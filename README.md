# yaml_json_tokenization_analyzer

Created by [Wayne Workman](https://github.com/wayneworkman)

[![Blog](https://img.shields.io/badge/Blog-wayne.theworkmans.us-blue)](https://wayne.theworkmans.us/)
[![GitHub](https://img.shields.io/badge/GitHub-wayneworkman-181717?logo=github)](https://github.com/wayneworkman)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Wayne_Workman-0077B5?logo=linkedin)](https://www.linkedin.com/in/wayne-workman-a8b37b353/)

## About

So analyzer.py takes a github username as a positional argument and pulls that users last 90 days of events from the GitHub API. It then stores all that data as both JSON and YAML formats and uses tiktoken to tokenize and count the tokens for each format. This shows YAML uses significantly less tokens than JSON for the exact same data.

The script also creates "reduced" versions that strip out all the unnecessary fields that GitHub returns. Like seriously, do you need 47 different API endpoint URLs for every single event? Or gravatar IDs? Or archived flags? Probably not. The reduced versions keep just the meaningful stuff like commit messages, PR titles, issue comments, repo names, basically what someone actually did rather than all the metadata about how to query more metadata.

After removing all that extra junk the token count drops by something like 80 percent. we're talking about going from 17,000 tokens down to 3,000 tokens for the same actual information.

Why should anyone care about token counts? If your using any of the paid LLMs like Nova, Sonnet, Opus, Llama, GPT-4, Claude, they all charge by tokens for both input and output so if your sending them JSON instead of YAML and not stripping out the useless fields your basically throwing money away on every single API call you make.

The script outputs nice ASCII tables using tabulate to show all the comparisons and also writes four files (raw_json.json, raw_yaml.yaml, reduced_json.json, reduced_yaml.yaml) so you can actually see what data we're talking about.

I built this because I wanted to see the token difference between formats but it ended up being way more dramatic than expected. Also made me realize how much completely redundant data these APIs return.

## Example Output

```
============================================================
GitHub Events Tokenization Analysis
User: wayneworkman | Events: 35
============================================================

TOKEN COUNT COMPARISON
+--------------------+---------------+---------------+--------------+----------------+
| Dataset            | JSON Tokens   | YAML Tokens   | Difference   | YAML vs JSON   |
+====================+===============+===============+==============+================+
| Raw (All Fields)   | 17,713        | 15,198        | -2,515       | -14.2%         |
+--------------------+---------------+---------------+--------------+----------------+
| Reduced (Filtered) | 3,397         | 2,625         | -772         | -22.7%         |
+--------------------+---------------+---------------+--------------+----------------+

DATA REDUCTION ANALYSIS
+----------+-------------------+------------------+----------------+---------------+
| Format   | Original Tokens   | Reduced Tokens   | Tokens Saved   | Reduction %   |
+==========+===================+==================+================+===============+
| JSON     | 17,713            | 3,397            | 14,316         | 80.8%         |
+----------+-------------------+------------------+----------------+---------------+
| YAML     | 15,198            | 2,625            | 12,573         | 82.7%         |
+----------+-------------------+------------------+----------------+---------------+

FILES CREATED
+-------------------+---------------+----------------------------------------+
| File              | Token Count   | Description                            |
+===================+===============+========================================+
| raw_json.json     | 17,713        | Full GitHub events data in JSON format |
+-------------------+---------------+----------------------------------------+
| raw_yaml.yaml     | 15,198        | Full GitHub events data in YAML format |
+-------------------+---------------+----------------------------------------+
| reduced_json.json | 3,397         | Filtered events data in JSON format    |
+-------------------+---------------+----------------------------------------+
| reduced_yaml.yaml | 2,625         | Filtered events data in YAML format    |
+-------------------+---------------+----------------------------------------+
```