# CIViC MCP Server

MCP-supported Chatbot for CIViC users: https://civicdb.org/mcp-chat

Preprint: https://www.biorxiv.org/content/10.1101/2025.10.13.682185v1

This is a Cloudflare Workers-based Model Context Protocol (MCP) server that provides tools for querying the CIViC (Clinical Interpretation of Variants in Cancer) API. 

The CIViC database is a crowd-sourced repository of clinical interpretations of cancer variants. This MCP server enables structured queries and data analysis of cancer genomics information through natural language interactions with AI assistants.

Python 3.11.7
```bash
pip install -r requirements.txt
```

## Directly Querying the MCP Server

The MCP Server can be called directly from the command line with the optional arguments disease and therapy to get CIViC evidence items:

```bash
python MCP_query_evidence.py --mp "EGFR" --disease "Lung Non-small Cell Carcinoma" --therapy "Erlotinib"
```

## Locally Hosting The CIViC MCP Server

We provide an example of doing this with GPT4o-mini. A personal API key is required, update the variable OPENAI_API_KEY in local_hosting/message_MCP_CIViC.py

```bash
python local_hosting/message_MCP_CIViC.py --msg "What is the clinical significance of EGFR variants in CIViC?" 
```

## Adding as a Claude Connector

<img width="3217" height="1094" alt="Claude connector setup" src="https://github.com/user-attachments/assets/b939db57-4a59-4cdf-a38e-620d9516cee4" />

## Usage

The server provides two main tools:

1. **`get_variant_evidence`**: Return up to 50 evidence items for a CIViC molecular profile
2. **`get_variant_assertions`**: Return CIViC assertions for a molecular profile


## Evaluation Scripts

All evaluation files are located in eval_QA_experiment/


## License

MIT License with Academic Citation Requirement - see [LICENSE.md](LICENSE.md)
