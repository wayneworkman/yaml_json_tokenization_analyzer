# yaml_json_tokenization_analyzer

Created by [Wayne Workman](https://github.com/wayneworkman)

[![Blog](https://img.shields.io/badge/Blog-wayne.theworkmans.us-blue)](https://wayne.theworkmans.us/)
[![GitHub](https://img.shields.io/badge/GitHub-wayneworkman-181717?logo=github)](https://github.com/wayneworkman)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Wayne_Workman-0077B5?logo=linkedin)](https://www.linkedin.com/in/wayne-workman-a8b37b353/)

## About

So analyzer.py takes a github username as a positional argument and pulls that users last 90 days of public events from the GitHub API. It then does two things, first if you have AWS access it tests how well different AI models can understand the data in both JSON and YAML formats. Second it measures token counts to see which format is more efficient.

My first round at this, I made the mistake of pretty-printing the JSON, and that inflated the JSON token counts. [@satwareAG-ironMike](https://github.com/satwareAG-ironMike) called this out in issue [#1](https://github.com/wayneworkman/yaml_json_tokenization_analyzer/issues/1), thank you Mike! To accurately compare JSON to YAML, the JSON has to be compact/minimized. With these corrections, the figures show YAML actually uses MORE tokens than JSON, not less. We're talking about 6-10% more tokens for the exact same data. I built this thinking YAML would be way more efficient but turns out I was totally wrong.

The script tests 6 different models (Nova Lite/Micro/Premier/Pro, Claude Sonnet, and Claude Opus) by asking them 5 simple questions about the GitHub data:

1. How many total events are there? (just count them)
2. What's the most common event type and how many times does it occur? 
3. How many PushEvents have more than 2 commits?
4. How many different event types are there total?
5. List all event types and their counts in descending order

These arent trick questions. The script literally calculates the right answers before asking the models. But most models completely fail at basic counting. Like they can't even count 32 events correctly. Claude Sonnet absolutely crushes it with 93.3% accuracy on JSON, but drops to 76.7% on YAML. Meanwhile Nova Pro somehow does better on YAML (80%) than JSON (56.7%).

What really blew my mind is Claude Sonnet 4 outperformed Claude Opus 4.1 in both JSON and YAML. For JSON it wasnt even close, Sonnet destroyed Opus. This is wild because Opus is supposed to be far more advanced than Sonnet. Also certain AWS Nova models actually work better with YAML than JSON. Nova Pro gets 80% accuracy on YAML but only 56.7% on JSON, and Nova Lite hits 56.7% on YAML which is just as good as Nova Pro on JSON. Thats quite striking when you think about it.

The script also creates "reduced" versions that strip out all the unnecessary fields that GitHub returns. Like seriously, do you need 47 different API endpoint URLs for every single event? Or gravatar IDs? Or archived flags? Probably not. The reduced versions keep just the meaningful stuff like commit messages, PR titles, issue comments, repo names, basically what someone actually did rather than all the metadata about how to query more metadata.

After removing all that extra junk the token count drops by something like 80 percent. we're talking about going from 40,000+ tokens down to 3,500 tokens for the same actual information. THAT'S where you save money on API calls, not by switching formats, but by not sending garbage data in the first place.

Why should anyone care about this? If your using any of the paid LLMs they all charge by tokens for both input and output. But more importantly if your asking models to analyze data, the format matters way more than I expected. JSON seems easier for Claude models to parse accurately as well as using slightly fewer tokens, while some Nova models perform better with YAML.

The script outputs nice ASCII tables using tabulate to show all the comparisons and also writes the prompts, responses, and data files so you can see exactly what the models got wrong. It uses the AWS Bedrock Converse API if you have AWS creds (because it returns exact token counts), otherwise falls back to tiktoken for basic counting.

I built this because I wanted to see the token difference between formats but ended up discovering that model accuracy varies WILDLY between formats. Also made me realize these models really struggle with what should be tirvial tasks.

The takeaway from all this testing is that keeping token costs down by using JSON is not the only consideration. You need to see how different models perform with different kinds of input and different kinds of output requirements, then pick whatever performs better while also being concious of costs. Claude Sonnet 4 is far more capable than I once thought, and I'm really floored that it outperformed Claude Opus.

## Example Output (snipped)

```
======================================================================
GitHub Events Tokenization Analysis
User: wayneworkman | Events: 32
======================================================================

Using AWS Bedrock Converse API (region: us-east-2)
======================================================================


================================================================================
COMPREHENSIVE MODEL TESTING
================================================================================

TOKEN COUNT COMPARISON ACROSS MODELS
+--------------------------+------------+------------+------------+------------+----------------------+----------------------+
| Model                    | Raw JSON   | Raw YAML   | Red JSON   | Red YAML   | JSON vs YAML (Raw)   | JSON vs YAML (Red)   |
+==========================+============+============+============+============+======================+======================+
| nova-lite                | 43,250     | 45,915     | 3,540      | 3,834      | +6.2%                | +8.3%                |
+--------------------------+------------+------------+------------+------------+----------------------+----------------------+
| nova-micro               | 43,762     | 46,427     | 3,540      | 3,834      | +6.1%                | +8.3%                |
+--------------------------+------------+------------+------------+------------+----------------------+----------------------+
| nova-premier             | 43,336     | 46,679     | 3,567      | 3,943      | +7.7%                | +10.5%               |
+--------------------------+------------+------------+------------+------------+----------------------+----------------------+
| nova-pro                 | 43,250     | 45,915     | 3,540      | 3,834      | +6.2%                | +8.3%                |
+--------------------------+------------+------------+------------+------------+----------------------+----------------------+
| claude-sonnet-4-20250514 | 37,911     | 40,385     | 3,579      | 3,815      | +6.5%                | +6.6%                |
+--------------------------+------------+------------+------------+------------+----------------------+----------------------+
| claude-opus-4-1-20250805 | 37,911     | 40,385     | 3,579      | 3,815      | +6.5%                | +6.6%                |
+--------------------------+------------+------------+------------+------------+----------------------+----------------------+

COMPREHENSION TEST RESULTS
+--------------------------+--------------+-----------------+--------------+-----------------+---------------+---------------+
| Model                    | JSON Parse   | JSON Accuracy   | YAML Parse   | YAML Accuracy   | JSON Tokens   | YAML Tokens   |
+==========================+==============+=================+==============+=================+===============+===============+
| nova-lite                | ✓            | 40.0%           | ✓            | 56.7%           | 3,779/102     | 4,074/135     |
+--------------------------+--------------+-----------------+--------------+-----------------+---------------+---------------+
| nova-micro               | ✓            | 43.3%           | ✓            | 43.3%           | 3,779/125     | 4,074/113     |
+--------------------------+--------------+-----------------+--------------+-----------------+---------------+---------------+
| nova-premier             | ✓            | 63.3%           | ✓            | 63.3%           | 3,818/111     | 4,197/123     |
+--------------------------+--------------+-----------------+--------------+-----------------+---------------+---------------+
| nova-pro                 | ✓            | 56.7%           | ✓            | 80.0%           | 3,779/126     | 4,074/113     |
+--------------------------+--------------+-----------------+--------------+-----------------+---------------+---------------+
| claude-sonnet-4-20250514 | ✓            | 93.3%           | ✓            | 76.7%           | 3,830/136     | 4,068/115     |
+--------------------------+--------------+-----------------+--------------+-----------------+---------------+---------------+
| claude-opus-4-1-20250805 | ✓            | 73.3%           | ✓            | 66.7%           | 3,830/136     | 4,068/115     |
+--------------------------+--------------+-----------------+--------------+-----------------+---------------+---------------+

DETAILED ACCURACY BREAKDOWN
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| Model                    | Format   | Q1: Total   | Q2: Event Type   | Q3: Push >2   | Q4: Event Types   |   Q5: Distribution | Overall   |
+==========================+==========+=============+==================+===============+===================+====================+===========+
| nova-lite                | JSON     | ✗           | ✓                | ✗             | ✗                 |                1.4 | 40.0%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| nova-lite                | YAML     | ✗           | ✓                | ✗             | ✓                 |                1.4 | 56.7%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| nova-micro               | JSON     | ✗           | ✓                | ✗             | ✗                 |                1.6 | 43.3%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| nova-micro               | YAML     | ✗           | ✓                | ✗             | ✗                 |                1.6 | 43.3%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| nova-premier             | JSON     | ✗           | ✓                | ✓             | ✗                 |                1.8 | 63.3%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| nova-premier             | YAML     | ✗           | ✓                | ✗             | ✓                 |                1.8 | 63.3%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| nova-pro                 | JSON     | ✗           | ✓                | ✗             | ✓                 |                1.4 | 56.7%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| nova-pro                 | YAML     | ✗           | ✓                | ✓             | ✓                 |                1.8 | 80.0%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| claude-sonnet-4-20250514 | JSON     | ✓           | ✓                | ✓             | ✓                 |                1.6 | 93.3%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| claude-sonnet-4-20250514 | YAML     | ✓           | ✓                | ✗             | ✓                 |                1.6 | 76.7%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| claude-opus-4-1-20250805 | JSON     | ✗           | ✓                | ✓             | ✓                 |                1.4 | 73.3%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+
| claude-opus-4-1-20250805 | YAML     | ✗           | ✓                | ✓             | ✓                 |                1   | 66.7%     |
+--------------------------+----------+-------------+------------------+---------------+-------------------+--------------------+-----------+


```