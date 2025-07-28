# CIViC MCP Server

This is a Cloudflare Workers-based Model Context Protocol (MCP) server that provides tools for querying the CIViC (Clinical Interpretation of Variants in Cancer) API. 

The CIViC database is a crowd-sourced repository of clinical interpretations of cancer variants. This MCP server enables structured queries and data analysis of cancer genomics information through natural language interactions with AI assistants.

### Directly Querying the MCP Server

The MCP Server can be called directly from the command line with the optional arguments disease and therapy to get CIViC evidence items:

```bash
pip install "mcp[cli]"
```

```bash
python MCP_query_evidence.py --mp "EGFR" --disease "Lung Non-small Cell Carcinoma" --therapy "Erlotinib"
```

### Locally Hosting The CIViC MCP Server

We provide an example of doing this with GPT4o. A personal API key is required.


### Using With Claude Desktop

Install Node.js (https://nodejs.org/)

Click "LTS" (Recommended for Most Users) — this gives you Node.js and npx
Download and install it like any normal app

Once installed:
On Windows: Open “Command Prompt” or “PowerShell”
On macOS: Open “Terminal”

Then run:
```bash
node -v
npx -v
```

Confirm that both give versions.

Add this configuration to your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "civic-mcp-server": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://civic-mcp-server.larscivic.workers.dev/sse"
      ]
    }
  }
}
```


## Usage

Once configured, restart Claude Desktop. The server provides two main tools:

1. **`get_variant_evidence`**: Return up to 10 evidence items for a CIViC molecular profile
2. **`get_variant_assertions`**: Return CIViC assertions for a molecular profile

## License

MIT License with Academic Citation Requirement - see [LICENSE.md](LICENSE.md)
