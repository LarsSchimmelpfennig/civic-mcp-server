#!/usr/bin/env python
"""

Call the `get_variant_evidence` tool on the CIViC MCP server from the command line.

Examples
--------
python MCP_query_evidence.py --mp "EGFR" \
                         --disease "Lung Non-small Cell Carcinoma" \
                         --therapy "Erlotinib"
"""
import argparse
import asyncio
import logging
from typing import Dict
import os

from mcp import ClientSession
from mcp.client.sse import sse_client 
from mcp.client.streamable_http import streamablehttp_client

#MCP_SSE_URL = "https://civic-mcp-server.larscivic.workers.dev/sse"
MCP_HTTP_URL = "https://civic-mcp-server.larscivic.workers.dev/mcp"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# -----------------------------------------------------------------------------

async def query_evidence(mp: str, disease: str | None, therapy: str | None) -> None:
    """Fetch evidence from the MCP server for the given molecular profile."""
    logging.info("Connecting to %s …", MCP_HTTP_URL)

    async with streamablehttp_client(url=MCP_HTTP_URL) as (read, write, _), \
               ClientSession(read, write) as session:
        # Handshake
        logging.info("→ initialize()")
        await session.initialize()

        # Compose argument dict (omit keys that are None)
        args: Dict[str, str] = {"molecularProfileName": mp}
        if disease:
            args["diseaseName"] = disease
        if therapy:
            args["therapyName"] = therapy

        logging.info("→ call_tool(get_variant_evidence)")
        result = await session.call_tool(
            name="get_variant_evidence",
            arguments=args,
        )

        # Print the first content block (usually JSON text)
        if result.content:
            print(result.content[0].text)
        else:
            logging.warning("No content returned by the tool.")

# async def query_evidence(mp: str, disease: str | None, therapy: str | None) -> None:
#     """Fetch evidence from the MCP server for the given molecular profile."""
#     logging.info("Connecting to %s …", MCP_SSE_URL)

#     async with sse_client(url=MCP_SSE_URL) as (read, write), ClientSession(
#         read, write
#     ) as session:
#         # Handshake
#         logging.info("→ initialize()")
#         await session.initialize()

#         # Compose argument dict (omit keys that are None)
#         args: Dict[str, str] = {"molecularProfileName": mp}
#         if disease:
#             args["diseaseName"] = disease
#         if therapy:
#             args["therapyName"] = therapy

#         logging.info("→ call_tool(get_variant_evidence)")
#         result = await session.call_tool(
#             name="get_variant_evidence",
#             arguments=args,
#         )

#         # Print the first content block (usually JSON text)
#         if result.content:
#             print(result.content[0].text)
#         else:
#             logging.warning("No content returned by the tool.")

# # -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query CIViC MCP get_variant_evidence tool."
    )
    parser.add_argument(
        "--mp",
        required=True,
        help="Molecular profile name (required).",
        dest="mp",
    )
    parser.add_argument(
        "--disease",
        help="Disease name (optional).",
        dest="disease",
        default=None,
    )
    parser.add_argument(
        "--therapy",
        help="Therapy name (optional).",
        dest="therapy",
        default=None,
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(query_evidence(args.mp, args.disease, args.therapy))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Exiting…")
