#!/usr/bin/env python
"""
Fetch GitHub events for a user and store as JSON and YAML
"""

from github import Github
import os
import sys
import json
import yaml
import tiktoken
from tabulate import tabulate
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# AWS Bedrock Configuration
AWS_DEFAULT_REGION = 'us-east-2'

# Model list for testing.
BEDROCK_MODELS = [
    'us.amazon.nova-lite-v1:0',
    'us.amazon.nova-micro-v1:0',
    'us.amazon.nova-premier-v1:0',
    'us.amazon.nova-pro-v1:0',
    'global.anthropic.claude-sonnet-4-20250514-v1:0',
    'us.anthropic.claude-opus-4-1-20250805-v1:0'
]

def generate_comprehension_questions(events_data, data_format):
    """Generate comprehension questions based on the actual data"""

    # Analyze data for question generation
    total_events = len(events_data)
    event_types = {}
    repos = set()
    push_events = []
    total_commits = 0

    for event in events_data:
        event_type = event.get('type', '')
        event_types[event_type] = event_types.get(event_type, 0) + 1

        if event.get('repo'):
            repos.add(event['repo'])

        if event_type == 'PushEvent' and event.get('commits'):
            push_events.append(event)
            total_commits += len(event.get('commits', []))

    most_common_event = max(event_types.items(), key=lambda x: x[1])
    unique_repos = len(repos)

    format_instruction = f"Answer in valid {data_format.upper()} format" if data_format == 'json' else "Answer in valid YAML format"

    questions = [
        f"Q1: How many total events are in this dataset? {format_instruction}.",
        f"Q2: What is the most common event type and how many times does it occur? {format_instruction}.",
        f"Q3: How many PushEvents contain more than 2 commits? {format_instruction}.",
        f"Q4: How many different event types are there in total? {format_instruction}.",
        f"Q5: List all event types and their counts in descending order by count. {format_instruction}."
    ]

    # Calculate new Q3 and Q4 answers
    push_events_with_many_commits = 0
    for event in push_events:
        if len(event.get('commits', [])) > 2:
            push_events_with_many_commits += 1

    total_event_types = len(event_types)

    # Expected answers for verification
    expected_answers = {
        'total_events': total_events,
        'most_common_event_type': most_common_event[0],
        'most_common_event_count': most_common_event[1],
        'push_events_with_many_commits': push_events_with_many_commits,
        'total_event_types': total_event_types,
        'total_commits': total_commits,
        'event_type_counts': sorted(event_types.items(), key=lambda x: x[1], reverse=True)
    }

    return questions, expected_answers

def grade_model_answer(parsed_answer, expected_answers, data_format):
    """Grade model answer against expected values"""
    if not parsed_answer:
        return {'total_score': 0, 'max_score': 5, 'scores': {}, 'accuracy': 0.0}

    scores = {}

    # Helper function to extract numeric values from answer
    def extract_number(value):
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            # Try to extract number from string
            import re
            numbers = re.findall(r'\d+', value)
            if numbers:
                return int(numbers[0])
        return None

    # Helper function to extract string values
    def extract_string(value):
        if isinstance(value, str):
            return value.lower().strip()
        return str(value).lower().strip() if value else ""

    # Q1: Total events count
    try:
        if data_format == 'json':
            answer_total = extract_number(parsed_answer.get('total_events') or
                                        parsed_answer.get('q1') or
                                        parsed_answer.get('1'))
        else:  # yaml
            answer_total = extract_number(parsed_answer.get('total_events') or
                                        parsed_answer.get('q1') or
                                        parsed_answer.get(1))

        scores['q1_total_events'] = 1 if answer_total == expected_answers['total_events'] else 0
    except:
        scores['q1_total_events'] = 0

    # Q2: Most common event type
    try:
        if data_format == 'json':
            answer_event = extract_string(parsed_answer.get('most_common_event') or
                                        parsed_answer.get('most_common_event_type') or
                                        parsed_answer.get('q2') or
                                        parsed_answer.get('2'))
        else:  # yaml
            answer_event = extract_string(parsed_answer.get('most_common_event') or
                                        parsed_answer.get('most_common_event_type') or
                                        parsed_answer.get('q2') or
                                        parsed_answer.get(2))

        expected_event = expected_answers['most_common_event_type'].lower()
        scores['q2_most_common_event'] = 1 if expected_event in answer_event or answer_event in expected_event else 0
    except:
        scores['q2_most_common_event'] = 0

    # Q3: PushEvents with more than 2 commits
    try:
        if data_format == 'json':
            answer_push_many = extract_number(parsed_answer.get('push_events_with_many_commits') or
                                            parsed_answer.get('pushevents_with_many_commits') or
                                            parsed_answer.get('q3') or
                                            parsed_answer.get('3'))
        else:  # yaml
            answer_push_many = extract_number(parsed_answer.get('push_events_with_many_commits') or
                                            parsed_answer.get('pushevents_with_many_commits') or
                                            parsed_answer.get('q3') or
                                            parsed_answer.get(3))

        scores['q3_push_events_with_many_commits'] = 1 if answer_push_many == expected_answers['push_events_with_many_commits'] else 0
    except:
        scores['q3_push_events_with_many_commits'] = 0

    # Q4: Total number of different event types
    try:
        if data_format == 'json':
            answer_types = extract_number(parsed_answer.get('total_event_types') or
                                        parsed_answer.get('unique_event_types') or
                                        parsed_answer.get('q4') or
                                        parsed_answer.get('4'))
        else:  # yaml
            answer_types = extract_number(parsed_answer.get('total_event_types') or
                                        parsed_answer.get('unique_event_types') or
                                        parsed_answer.get('q4') or
                                        parsed_answer.get(4))

        scores['q4_total_event_types'] = 1 if answer_types == expected_answers['total_event_types'] else 0
    except:
        scores['q4_total_event_types'] = 0

    # Q5: Event type distribution (position + count accuracy)
    try:
        if data_format == 'json':
            answer_distribution = parsed_answer.get('event_type_counts') or parsed_answer.get('event_types') or parsed_answer.get('q5') or parsed_answer.get('5')
        else:  # yaml
            answer_distribution = parsed_answer.get('event_type_counts') or parsed_answer.get('event_types') or parsed_answer.get('q5') or parsed_answer.get(5)

        expected_items = expected_answers['event_type_counts']  # [('PushEvent', 24), ('WatchEvent', 4), ...]
        position_score = 0
        count_score = 0
        max_items = len(expected_items)

        if isinstance(answer_distribution, (list, dict)):
            if isinstance(answer_distribution, dict):
                # Convert dict to list of tuples sorted by count
                answer_items = sorted(answer_distribution.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
            else:
                answer_items = answer_distribution

            # Check each position
            for i, (expected_type, expected_count) in enumerate(expected_items):
                if i < len(answer_items):
                    # Extract event type and count from answer
                    if isinstance(answer_items[i], (list, tuple)) and len(answer_items[i]) >= 2:
                        answer_type = str(answer_items[i][0]).strip()
                        answer_count = extract_number(answer_items[i][1])
                    elif isinstance(answer_items[i], dict) and 'event_type' in answer_items[i] and 'count' in answer_items[i]:
                        answer_type = str(answer_items[i]['event_type']).strip()
                        answer_count = extract_number(answer_items[i]['count'])
                    else:
                        continue

                    # Position score: correct event type in correct position
                    if expected_type.lower() == answer_type.lower():
                        position_score += 1

                    # Count score: correct count for any event type that appears in expected
                    if answer_count == expected_count:
                        # Check if this event type exists in expected answers
                        expected_types = [item[0].lower() for item in expected_items]
                        if answer_type.lower() in expected_types:
                            count_score += 1

            # Handle ties: if two items have same count, either order should be acceptable
            # This is complex to implement perfectly, so we'll give partial credit for near-correct positioning

        # Normalize scores (position_score + count_score out of max_items * 2)
        total_possible = max_items * 2  # max_items for position + max_items for count
        total_score = position_score + count_score
        scores['q5_event_distribution'] = min(2.0, total_score / total_possible * 2) if total_possible > 0 else 0
    except:
        scores['q5_event_distribution'] = 0

    total_score = sum(scores.values())
    max_score = 6  # Q1(1) + Q2(1) + Q3(1) + Q4(1) + Q5(2) = 6 total points
    accuracy = (total_score / max_score) * 100 if max_score > 0 else 0

    return {
        'total_score': total_score,
        'max_score': max_score,
        'scores': scores,
        'accuracy': accuracy
    }

def parse_model_response(response_text, data_format, model_id):
    """Parse model response, handling code fences and format conversion

    Args:
        response_text: Raw response from model
        data_format: 'json' or 'yaml'
        model_id: Model identifier for debugging

    Returns:
        tuple: (parsed_data, success_flag)
    """
    parsed_text = response_text
    if '```' in response_text:
        parsed_text = response_text.split('```')[1]
        first_line = parsed_text.partition('\n')[0]
        if first_line == "json" or first_line == "yaml":
            parsed_text = "\n".join(parsed_text.split("\n")[1:])

    # Parse the text into a data structure
    try:
        if data_format == 'json':
            parsed_data = json.loads(parsed_text)
            return parsed_data, True
        else:  # yaml
            parsed_data = yaml.safe_load(parsed_text)
            return parsed_data, True
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"Parse error for {model_id} ({data_format}): {e}", file=sys.stderr)
        return None, False

def check_aws_credentials():
    """Check if AWS credentials are valid using STS get-caller-identity"""
    try:
        # Try to call STS to verify credentials
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()

        # If we get here, credentials are valid
        aws_region = os.environ.get('AWS_DEFAULT_REGION', AWS_DEFAULT_REGION)
        print(f"Valid AWS credentials detected (Account: {identity['Account']})")
        print(f"Using region: {aws_region}")
        return True, aws_region
    except Exception:
        print("AWS access is not detected (GetCallerIdentity failed), using tiktoken instead of AWS Bedrock Converse API.")
        return False, None

def get_bedrock_token_count(client, text, model_id):
    """Get token count using AWS Bedrock Converse API"""
    try:
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': text}]
                }
            ],
            inferenceConfig={
                'maxTokens': 10,  # Minimal response to save costs
                'temperature': 0.0
            }
        )
        return response['usage']['inputTokens']
    except Exception as e:
        print(f"Error getting token count from Bedrock model {model_id}: {e}", file=sys.stderr)
        return None

def test_model_comprehension(client, model_id, questions, data, data_format, expected_answers):
    """Test model comprehension with questions and validate answers"""
    try:
        # Create the prompt with questions and data
        questions_text = "\n".join(questions)
        format_name = data_format.upper()

        if data_format == 'json':
            prompt = f"""You must respond with ONLY valid JSON. Do not include any explanatory text, markdown formatting, or code blocks.

Answer these questions about the GitHub events data:

{questions_text}

Respond with a JSON object containing your answers. Use these exact keys:
- "total_events": number
- "most_common_event_type": string
- "most_common_event_count": number
- "push_events_with_many_commits": number
- "total_event_types": number
- "event_type_counts": array of [event_type, count] pairs sorted by count descending

Here is the JSON data to analyze:

{data}"""
        else:  # yaml
            prompt = f"""You must respond with ONLY valid YAML. Do not include any explanatory text, markdown formatting, or code blocks.

Answer these questions about the GitHub events data:

{questions_text}

Respond with a YAML document containing your answers. Use these exact keys:
- total_events: number
- most_common_event_type: string
- most_common_event_count: number
- push_events_with_many_commits: number
- total_event_types: number
- event_type_counts: array of [event_type, count] pairs sorted by count descending

Here is the YAML data to analyze:

{data}"""

        # Create prompts_and_responses directory if it doesn't exist
        os.makedirs('prompts_and_responses', exist_ok=True)

        # Save input prompt to file
        model_short_name = model_id.split('.')[-1].replace('-v1:0', '')
        prompt_filename = f"prompts_and_responses/{model_short_name}_{data_format}_prompt.txt"
        with open(prompt_filename, 'w', encoding='utf-8') as f:
            f.write(f"MODEL: {model_id}\n")
            f.write(f"FORMAT: {data_format.upper()}\n")
            f.write(f"{'='*80}\n\n")
            f.write(prompt)

        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': prompt}]
                }
            ],
            inferenceConfig={
                'maxTokens': 1000,
                'temperature': 0.0
            }
        )

        answer_text = response['output']['message']['content'][0]['text']
        input_tokens = response['usage']['inputTokens']
        output_tokens = response['usage']['outputTokens']

        # Parse the response immediately using dedicated function
        parsed_answer, parse_success = parse_model_response(answer_text, data_format, model_id)

        # Save both raw and parsed response to file
        response_filename = f"prompts_and_responses/{model_short_name}_{data_format}_response.txt"
        with open(response_filename, 'w', encoding='utf-8') as f:
            f.write(f"MODEL: {model_id}\n")
            f.write(f"FORMAT: {data_format.upper()}\n")
            f.write(f"INPUT TOKENS: {input_tokens}\n")
            f.write(f"OUTPUT TOKENS: {output_tokens}\n")
            f.write(f"{'='*80}\n\n")
            f.write("RAW RESPONSE:\n")
            f.write(answer_text)
            f.write(f"\n\n{'='*80}\n\n")
            f.write("PARSED RESPONSE:\n")
            if parsed_answer:
                if data_format == 'json':
                    f.write(json.dumps(parsed_answer, indent=2))
                else:
                    f.write(yaml.dump(parsed_answer))
            else:
                f.write("FAILED TO PARSE")

        # Grade the answer if parsing was successful
        grading_results = grade_model_answer(parsed_answer, expected_answers, data_format)

        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'answer_text': answer_text,
            'parse_success': parse_success,
            'parsed_answer': parsed_answer,
            'grading_results': grading_results
        }

    except Exception as e:
        import traceback
        print(f"Error testing model {model_id}: {e}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        return None

def test_all_models_comprehensive(client, events_data, raw_json, raw_yaml, reduced_json, reduced_yaml):
    """Test all models with both JSON and YAML data"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL TESTING")
    print(f"{'='*80}\n")

    results = {}

    # Generate questions for reduced data (more manageable)
    questions_json, expected_answers_json = generate_comprehension_questions(
        json.loads(reduced_json) if isinstance(reduced_json, str) else reduced_json, 'json'
    )
    questions_yaml, expected_answers_yaml = generate_comprehension_questions(
        yaml.safe_load(reduced_yaml) if isinstance(reduced_yaml, str) else reduced_yaml, 'yaml'
    )

    for model_id in BEDROCK_MODELS:
        model_results = {
            'raw_json_tokens': None,
            'raw_yaml_tokens': None,
            'reduced_json_tokens': None,
            'reduced_yaml_tokens': None,
            'json_comprehension': None,
            'yaml_comprehension': None
        }

        # Get token counts for all data formats
        model_results['raw_json_tokens'] = get_bedrock_token_count(client, raw_json, model_id)
        model_results['raw_yaml_tokens'] = get_bedrock_token_count(client, raw_yaml, model_id)
        model_results['reduced_json_tokens'] = get_bedrock_token_count(client, reduced_json, model_id)
        model_results['reduced_yaml_tokens'] = get_bedrock_token_count(client, reduced_yaml, model_id)

        # Test comprehension with reduced data
        model_results['json_comprehension'] = test_model_comprehension(
            client, model_id, questions_json, reduced_json, 'json', expected_answers_json
        )
        model_results['yaml_comprehension'] = test_model_comprehension(
            client, model_id, questions_yaml, reduced_yaml, 'yaml', expected_answers_yaml
        )

        results[model_id] = model_results

    return results, expected_answers_json

def display_comprehensive_results(results, expected_answers):
    """Display comprehensive model testing results"""

    # Token count comparison table
    print("TOKEN COUNT COMPARISON ACROSS MODELS")
    token_headers = ["Model", "Raw JSON", "Raw YAML", "Red JSON", "Red YAML", "JSON vs YAML (Raw)", "JSON vs YAML (Red)"]
    token_data = []

    for model_id, model_results in results.items():
        model_name = model_id.split('.')[-1].replace('-v1:0', '')  # Shorten model name

        raw_json = model_results['raw_json_tokens']
        raw_yaml = model_results['raw_yaml_tokens']
        red_json = model_results['reduced_json_tokens']
        red_yaml = model_results['reduced_yaml_tokens']

        # Calculate percentage differences
        if raw_yaml and raw_json:
            raw_pct = ((raw_yaml - raw_json) / raw_json) * 100
            raw_diff = f"+{raw_pct:.1f}%" if raw_pct > 0 else f"{raw_pct:.1f}%"
        else:
            raw_diff = "N/A"

        if red_yaml and red_json:
            red_pct = ((red_yaml - red_json) / red_json) * 100
            red_diff = f"+{red_pct:.1f}%" if red_pct > 0 else f"{red_pct:.1f}%"
        else:
            red_diff = "N/A"

        token_data.append([
            model_name,
            f"{raw_json:,}" if raw_json else "FAIL",
            f"{raw_yaml:,}" if raw_yaml else "FAIL",
            f"{red_json:,}" if red_json else "FAIL",
            f"{red_yaml:,}" if red_yaml else "FAIL",
            raw_diff,
            red_diff
        ])

    print(tabulate(token_data, headers=token_headers, tablefmt="grid"))
    print()

    # Comprehension test results with accuracy scores
    print("COMPREHENSION TEST RESULTS")
    comp_headers = ["Model", "JSON Parse", "JSON Accuracy", "YAML Parse", "YAML Accuracy", "JSON Tokens", "YAML Tokens"]
    comp_data = []

    for model_id, model_results in results.items():
        model_name = model_id.split('.')[-1].replace('-v1:0', '')

        json_comp = model_results['json_comprehension']
        yaml_comp = model_results['yaml_comprehension']

        # Parse success indicators
        json_parse = "✓" if json_comp and json_comp['parse_success'] else "✗"
        yaml_parse = "✓" if yaml_comp and yaml_comp['parse_success'] else "✗"

        # Accuracy scores
        json_accuracy = f"{json_comp['grading_results']['accuracy']:.1f}%" if json_comp and 'grading_results' in json_comp else "0.0%"
        yaml_accuracy = f"{yaml_comp['grading_results']['accuracy']:.1f}%" if yaml_comp and 'grading_results' in yaml_comp else "0.0%"

        # Token counts (combined in/out)
        json_tokens = f"{json_comp['input_tokens']:,}/{json_comp['output_tokens']:,}" if json_comp and 'input_tokens' in json_comp else "FAIL"
        yaml_tokens = f"{yaml_comp['input_tokens']:,}/{yaml_comp['output_tokens']:,}" if yaml_comp and 'input_tokens' in yaml_comp else "FAIL"

        comp_data.append([
            model_name,
            json_parse,
            json_accuracy,
            yaml_parse,
            yaml_accuracy,
            json_tokens,
            yaml_tokens
        ])

    print(tabulate(comp_data, headers=comp_headers, tablefmt="grid"))
    print()

    # Detailed accuracy breakdown
    print("DETAILED ACCURACY BREAKDOWN")
    detail_headers = ["Model", "Format", "Q1: Total", "Q2: Event Type", "Q3: Push >2", "Q4: Event Types", "Q5: Distribution", "Overall"]
    detail_data = []

    for model_id, model_results in results.items():
        model_name = model_id.split('.')[-1].replace('-v1:0', '')

        for format_name, comp_key in [("JSON", "json_comprehension"), ("YAML", "yaml_comprehension")]:
            comp = model_results[comp_key]
            if comp and 'grading_results' in comp:
                scores = comp['grading_results']['scores']
                detail_data.append([
                    model_name,
                    format_name,
                    "✓" if scores.get('q1_total_events', 0) == 1 else "✗",
                    "✓" if scores.get('q2_most_common_event', 0) == 1 else "✗",
                    "✓" if scores.get('q3_push_events_with_many_commits', 0) == 1 else "✗",
                    "✓" if scores.get('q4_total_event_types', 0) == 1 else "✗",
                    f"{scores.get('q5_event_distribution', 0):.1f}" if scores.get('q5_event_distribution', 0) < 2 else "✓",
                    f"{comp['grading_results']['accuracy']:.1f}%"
                ])
            else:
                detail_data.append([
                    model_name,
                    format_name,
                    "✗", "✗", "✗", "✗", "✗", "0.0%"
                ])

    print(tabulate(detail_data, headers=detail_headers, tablefmt="grid"))
    print()

    # Show expected answers for reference
    print("EXPECTED ANSWERS FOR VERIFICATION:")
    print(f"  • Total events: {expected_answers['total_events']}")
    print(f"  • Most common event: {expected_answers['most_common_event_type']} ({expected_answers['most_common_event_count']} times)")
    print(f"  • PushEvents with >2 commits: {expected_answers['push_events_with_many_commits']}")
    print(f"  • Total event types: {expected_answers['total_event_types']}")
    print(f"  • Event type distribution: {expected_answers['event_type_counts']}")
    print()

def main():
    # Check for username argument
    if len(sys.argv) != 2:
        print("GitHub Events Fetcher")
        print("=" * 30)
        print("Usage: python analyzer.py <github_username>")
        print("\nExample:")
        print("  python analyzer.py wayneworkman")
        print("\nOptional:")
        print("  Set GITHUB_TOKEN environment variable for higher API rate limits")
        print("  Set AWS credentials to test all models, otherwise uses tiktoken")
        sys.exit(1)

    username = sys.argv[1]

    # Check AWS credentials first
    has_valid_aws_creds, aws_region = check_aws_credentials()

    # Get GitHub token from environment variable (optional)
    token = os.environ.get("GITHUB_TOKEN", None)

    if token:
        g = Github(token)
    else:
        g = Github()

    try:
        # Get user
        user = g.get_user(username)

        # Get user events (public events)
        events = user.get_public_events()

        # Convert events to list of dictionaries (raw data)
        events_data = []
        for event in events:
            # Get the raw data from the event object
            event_dict = {
                'id': event.id,
                'type': event.type,
                'actor': {
                    'id': event.actor.id if event.actor else None,
                    'login': event.actor.login if event.actor else None,
                    'display_login': event.actor.display_login if event.actor else None,
                    'gravatar_id': event.actor.gravatar_id if event.actor else None,
                    'url': event.actor.url if event.actor else None,
                    'avatar_url': event.actor.avatar_url if event.actor else None
                },
                'repo': {
                    'id': event.repo.id if event.repo else None,
                    'name': event.repo.name if event.repo else None,
                    'url': event.repo.url if event.repo else None
                },
                'payload': event.payload,
                'public': event.public,
                'created_at': event.created_at.isoformat() if event.created_at else None,
                'org': {
                    'id': event.org.id if event.org else None,
                    'login': event.org.login if event.org else None,
                    'gravatar_id': event.org.gravatar_id if event.org else None,
                    'url': event.org.url if event.org else None,
                    'avatar_url': event.org.avatar_url if event.org else None
                } if event.org else None
            }
            events_data.append(event_dict)

        # Store as raw JSON dumped text (minimized for API calls)
        raw_json = json.dumps(events_data, separators=(',', ':'), default=str)

        # Store as raw YAML dumped text (minimized for API calls)
        raw_yaml = yaml.dump(events_data, default_flow_style=False, allow_unicode=True, width=float('inf'))

        # Create raw_data directory if it doesn't exist
        os.makedirs('raw_data', exist_ok=True)

        # Write raw JSON and YAML to files (pretty-printed for readability)
        with open("raw_data/raw_json.json", "w") as f:
            f.write(json.dumps(events_data, indent=2, default=str))

        with open("raw_data/raw_yaml.yaml", "w") as f:
            f.write(yaml.dump(events_data, default_flow_style=False, allow_unicode=True))

        # Create reduced versions focusing on meaningful fields
        reduced_events_data = []
        for event in events_data:
            reduced_event = {
                'type': event['type'],
                'created_at': event['created_at'],
                'repo': event['repo']['name'] if event['repo'] else None,
            }

            # Add type-specific meaningful fields
            payload = event.get('payload', {})

            if event['type'] == 'PushEvent':
                reduced_event['commits'] = [
                    {'message': c['message'], 'author': c.get('author', {}).get('name')}
                    for c in payload.get('commits', [])[:3]  # Keep first 3 commits
                ]
                reduced_event['branch'] = payload.get('ref', '').replace('refs/heads/', '')

            elif event['type'] == 'PullRequestEvent':
                pr = payload.get('pull_request', {})
                reduced_event['action'] = payload.get('action')
                reduced_event['pr_title'] = pr.get('title')
                reduced_event['pr_state'] = pr.get('state')
                reduced_event['pr_merged'] = pr.get('merged')

            elif event['type'] == 'IssuesEvent':
                issue = payload.get('issue', {})
                reduced_event['action'] = payload.get('action')
                reduced_event['issue_title'] = issue.get('title')
                reduced_event['issue_state'] = issue.get('state')

            elif event['type'] == 'IssueCommentEvent':
                issue = payload.get('issue', {})
                comment = payload.get('comment', {})
                reduced_event['issue_title'] = issue.get('title')
                reduced_event['comment_preview'] = comment.get('body', '')[:200]

            elif event['type'] == 'CreateEvent':
                reduced_event['ref'] = payload.get('ref')
                reduced_event['ref_type'] = payload.get('ref_type')

            elif event['type'] == 'DeleteEvent':
                reduced_event['ref'] = payload.get('ref')
                reduced_event['ref_type'] = payload.get('ref_type')

            elif event['type'] == 'ForkEvent':
                forkee = payload.get('forkee', {})
                reduced_event['fork_name'] = forkee.get('full_name')
                reduced_event['fork_description'] = forkee.get('description')

            elif event['type'] == 'WatchEvent':
                reduced_event['action'] = payload.get('action')

            elif event['type'] == 'ReleaseEvent':
                release = payload.get('release', {})
                reduced_event['action'] = payload.get('action')
                reduced_event['tag'] = release.get('tag_name')
                reduced_event['release_name'] = release.get('name')

            reduced_events_data.append(reduced_event)

        # Store as reduced JSON dumped text (minimized for API calls)
        reduced_json = json.dumps(reduced_events_data, separators=(',', ':'), default=str)

        # Store as reduced YAML dumped text (minimized for API calls)
        reduced_yaml = yaml.dump(reduced_events_data, default_flow_style=False, allow_unicode=True, width=float('inf'))

        # Create reduced_data directory if it doesn't exist
        os.makedirs('reduced_data', exist_ok=True)

        # Write reduced JSON and YAML to files (pretty-printed for readability)
        with open("reduced_data/reduced_json.json", "w") as f:
            f.write(json.dumps(reduced_events_data, indent=2, default=str))

        with open("reduced_data/reduced_yaml.yaml", "w") as f:
            f.write(yaml.dump(reduced_events_data, default_flow_style=False, allow_unicode=True))

        # Print header
        print(f"\n{'='*70}")
        print(f"GitHub Events Tokenization Analysis")
        print(f"User: {username} | Events: {len(events_data)}")
        print(f"{'='*70}\n")

        # Two completely separate paths based on AWS credentials
        if has_valid_aws_creds:
            # AWS PATH - Full model testing with Bedrock
            print(f"Using AWS Bedrock Converse API (region: {aws_region})")
            print(f"{'='*70}\n")

            # Create Bedrock client
            bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)

            # Run comprehensive model testing
            comprehensive_results, expected_answers = test_all_models_comprehensive(
                bedrock_client, events_data, raw_json, raw_yaml, reduced_json, reduced_yaml
            )
            display_comprehensive_results(comprehensive_results, expected_answers)

        else:
            # TIKTOKEN PATH - Simple token counting only
            print("Using tiktoken (cl100k_base) for token counting")
            print(f"{'='*70}\n")

            # Initialize tiktoken encoder
            enc = tiktoken.get_encoding("cl100k_base")

            # Get token counts
            json_token_count = len(enc.encode(raw_json))
            yaml_token_count = len(enc.encode(raw_yaml))
            reduced_json_token_count = len(enc.encode(reduced_json))
            reduced_yaml_token_count = len(enc.encode(reduced_yaml))

            # Calculate differences and percentages
            raw_diff = yaml_token_count - json_token_count
            raw_pct = (yaml_token_count / json_token_count - 1) * 100 if json_token_count > 0 else 0

            reduced_diff = reduced_yaml_token_count - reduced_json_token_count
            reduced_pct = (reduced_yaml_token_count / reduced_json_token_count - 1) * 100 if reduced_json_token_count > 0 else 0

            json_reduction_pct = (1 - reduced_json_token_count / json_token_count) * 100 if json_token_count > 0 else 0
            yaml_reduction_pct = (1 - reduced_yaml_token_count / yaml_token_count) * 100 if yaml_token_count > 0 else 0

            # Token Count Comparison (using tiktoken)
            print("TOKEN COUNT COMPARISON")
            print()

            token_headers = ["Dataset", "JSON Tokens", "YAML Tokens", "JSON vs YAML (Diff)", "JSON vs YAML (%)"]
            token_data = [
                ["Raw (All Fields)",
                 f"{json_token_count:,}",
                 f"{yaml_token_count:,}",
                 f"+{yaml_token_count - json_token_count:,}" if yaml_token_count > json_token_count else f"{yaml_token_count - json_token_count:,}",
                 f"+{raw_pct:.1f}%" if raw_pct > 0 else f"{raw_pct:.1f}%"],
                ["Reduced (Filtered)",
                 f"{reduced_json_token_count:,}",
                 f"{reduced_yaml_token_count:,}",
                 f"+{reduced_yaml_token_count - reduced_json_token_count:,}" if reduced_yaml_token_count > reduced_json_token_count else f"{reduced_yaml_token_count - reduced_json_token_count:,}",
                 f"+{reduced_pct:.1f}%" if reduced_pct > 0 else f"{reduced_pct:.1f}%"]
            ]

            print(tabulate(token_data, headers=token_headers, tablefmt="grid"))
            print()

            # Data Reduction Table
            reduction_data = [
                ["JSON", f"{json_token_count:,}", f"{reduced_json_token_count:,}",
                 f"{json_token_count - reduced_json_token_count:,}", f"{json_reduction_pct:.1f}%"],
                ["YAML", f"{yaml_token_count:,}", f"{reduced_yaml_token_count:,}",
                 f"{yaml_token_count - reduced_yaml_token_count:,}", f"{yaml_reduction_pct:.1f}%"]
            ]
            reduction_headers = ["Format", "Original Tokens", "Reduced Tokens", "Tokens Saved", "Reduction %"]

            print("DATA REDUCTION ANALYSIS")
            print(tabulate(reduction_data, headers=reduction_headers, tablefmt="grid"))
            print()

            # Files Created Table
            files_data = [
                ["raw_data/raw_json.json", f"{json_token_count:,}", "Full GitHub events data in JSON format"],
                ["raw_data/raw_yaml.yaml", f"{yaml_token_count:,}", "Full GitHub events data in YAML format"],
                ["reduced_data/reduced_json.json", f"{reduced_json_token_count:,}", "Filtered events data in JSON format"],
                ["reduced_data/reduced_yaml.yaml", f"{reduced_yaml_token_count:,}", "Filtered events data in YAML format"]
            ]
            files_headers = ["File", "Token Count", "Description"]

            print("FILES CREATED")
            print(tabulate(files_data, headers=files_headers, tablefmt="grid"))
            print()


    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()