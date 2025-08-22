import asyncio, os, sys
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.mcp import MCPServerStdio, MCPServerStdioParams
import argparse
import logging
# logging.basicConfig(level=logging.DEBUG)

# class ToolInvocationFilter(logging.Filter):
#     def filter(self, record):
#         return (
#             record.levelname == "DEBUG"
#             and record.name == "openai.agents"
#             and record.getMessage().startswith("Invoking MCP tool")
#        )

# # Add the filter to the root logger or your desired handler
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# handler.addFilter(ToolInvocationFilter())

# ##Replace the root logger's handlers with your filtered handler
# logging.getLogger().handlers = [handler]
# logging.getLogger().setLevel(logging.DEBUG)

root = logging.getLogger()
root.handlers = []
root.setLevel(logging.DEBUG)

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
root.addHandler(stream)

# Often useful granular loggers:
logging.getLogger("openai.agents").setLevel(logging.DEBUG)
logging.getLogger("openai.agents.mcp").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)

import functools, sys
print = functools.partial(print, file=sys.stderr, flush=True) 


#OPENAI_API_KEY = 'INSERT API KEY'

OPENAI_API_KEY = os.getenv("openai_api_key")

server_params = {
    "command": "fastmcp",
    "args": ["run", "host_MCP_CIViC.py:mcp", "--transport", "stdio"],
    "env": {              # extra env vars get passed to the subprocess
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "MCP_DEBUG": "1",     # optional: turns on verbose fastmcp logging
    },
    "errlog": sys.stderr,     # <-- pipe server stderr straight to *this* terminal
}

srv = MCPServerStdio(name="CIViC tools", params=server_params)

async def main(msg):

    await srv.__aenter__()

    agent = Agent(
        name="CIViC-Assistant",
        instructions=(
            "Use the tools to answer oncology variant questions for the Clinical Interpretations of Variants in Cancer (CIViC) knowledgebase."
            "**Definition of each evidence type**\n"
            "Diagnostic: Evidence pertains to a variant’s impact on patient diagnosis (cancer subtype).\n"
            "Predictive: Evidence pertains to a variant’s effect on therapeutic response.\n"
            "Prognostic: Evidence pertains to a variant’s impact on disease progression, severity, or patient survival.\n"
            "Predisposing: Evidence pertains to a germline Molecular Profile’s role in conferring susceptibility to disease (including pathogenicity evaluations)\n"
            "Oncogenic: Evidence pertains to a somatic variant’s involvement in tumor pathogenesis as described by the Hallmarks of Cancer.\n"
            "Functional: Evidence pertains to a variant that alters biological function from the reference state.\n"
            "Call both get_variant_evidence and get_variant_assertions to understand clinical significance.\n"
            "IMPORTANT: When using information from a specific evidence item or assertion, cite it with the associated url."
        ),
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=AsyncOpenAI()
        ),
        model_settings=ModelSettings(temperature=0.2),
        mcp_servers=[srv],
    )

    result = await Runner.run(
        agent,
        msg
    )
    print('###LLM OUTPUT###')
    print(result.final_output)


    await srv.__aexit__(None, None, None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Host an LLM to interact with CIViC MCP.")

    parser.add_argument(
        "--msg",
        required=True,
        help="Message for LLM",
        dest="msg",
    )

    args = parser.parse_args()

    asyncio.run(main(args.msg))