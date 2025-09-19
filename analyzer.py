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

def main():
    # Check for username argument
    if len(sys.argv) != 2:
        print("GitHub Events Fetcher")
        print("=" * 30)
        print("Usage: python github_events.py <github_username>")
        print("\nExample:")
        print("  python github_events.py wayneworkman")
        print("\nOptional:")
        print("  Set GITHUB_TOKEN environment variable for higher API rate limits")
        sys.exit(1)

    username = sys.argv[1]

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

        # Store as raw JSON dumped text
        raw_json = json.dumps(events_data, indent=2, default=str)

        # Store as raw YAML dumped text
        raw_yaml = yaml.dump(events_data, default_flow_style=False, allow_unicode=True)

        # Write raw JSON and YAML to files
        with open("raw_json.json", "w") as f:
            f.write(raw_json)

        with open("raw_yaml.yaml", "w") as f:
            f.write(raw_yaml)

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

        # Store as reduced JSON dumped text
        reduced_json = json.dumps(reduced_events_data, indent=2, default=str)

        # Store as reduced YAML dumped text
        reduced_yaml = yaml.dump(reduced_events_data, default_flow_style=False, allow_unicode=True)

        # Write reduced JSON and YAML to files
        with open("reduced_json.json", "w") as f:
            f.write(reduced_json)

        with open("reduced_yaml.yaml", "w") as f:
            f.write(reduced_yaml)

        # Initialize tiktoken encoder
        enc = tiktoken.get_encoding("cl100k_base")

        # Get token counts for all versions
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

        # Print header
        print(f"\n{'='*60}")
        print(f"GitHub Events Tokenization Analysis")
        print(f"User: {username} | Events: {len(events_data)}")
        print(f"{'='*60}\n")

        # Token Count Comparison Table
        token_data = [
            ["Raw (All Fields)", f"{json_token_count:,}", f"{yaml_token_count:,}",
             f"{raw_diff:+,d}" if raw_diff != 0 else "0",
             f"{raw_pct:+.1f}%" if raw_diff != 0 else "0.0%"],
            ["Reduced (Filtered)", f"{reduced_json_token_count:,}", f"{reduced_yaml_token_count:,}",
             f"{reduced_diff:+,d}" if reduced_diff != 0 else "0",
             f"{reduced_pct:+.1f}%" if reduced_diff != 0 else "0.0%"]
        ]
        token_headers = ["Dataset", "JSON Tokens", "YAML Tokens", "Difference", "YAML vs JSON"]

        print("TOKEN COUNT COMPARISON")
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
            ["raw_json.json", f"{json_token_count:,}", "Full GitHub events data in JSON format"],
            ["raw_yaml.yaml", f"{yaml_token_count:,}", "Full GitHub events data in YAML format"],
            ["reduced_json.json", f"{reduced_json_token_count:,}", "Filtered events data in JSON format"],
            ["reduced_yaml.yaml", f"{reduced_yaml_token_count:,}", "Filtered events data in YAML format"]
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