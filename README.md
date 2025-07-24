# CIViC MCP Server

This is a Cloudflare Workers-based Model Context Protocol (MCP) server that provides tools for querying the CIViC (Clinical Interpretation of Variants in Cancer) API. 

The CIViC database is a crowd-sourced repository of clinical interpretations of cancer variants. This MCP server enables structured queries and data analysis of cancer genomics information through natural language interactions with AI assistants.

## Directly Querying the MCP Server

The MCP Server can be called directly from the command line with optional arguments disease and therapy to get CIViC evidence items:

```bash
pip install "mcp[cli]"
```

```bash
python MCP_query_evidence.py --mp "molecularProfileNmae" --disease "diseaseName" --therapy "therapyName"
```

## To Host Your Own Version: Installation & Configuration

### Prerequisites
- A Cloudflare account
- Wrangler CLI installed
- Claude Desktop app

### Deploy to Cloudflare Workers

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd civic-mcp-server
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Deploy to Cloudflare Workers:
   ```bash
   npm run deploy
   ```

4. After deployment, you'll get a URL like: `https://civic-mcp-server.YOUR_SUBDOMAIN.workers.dev`

### Configure Claude Desktop

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

Replace `larscivic` with your actual Cloudflare Workers subdomain.

## Usage

Once configured, restart Claude Desktop. The server provides two main tools:

1. **`get_variant_evidence`**: Return up to 10 evidence items for a CIViC molecular profile
2. **`get_variant_assertions`**: Return CIViC assertions for a molecular profile

## License

MIT License with Academic Citation Requirement - see [LICENSE.md](LICENSE.md)
