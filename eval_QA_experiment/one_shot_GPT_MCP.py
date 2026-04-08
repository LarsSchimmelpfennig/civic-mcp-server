import asyncio
import os
import sys
import functools
import logging
import pandas as pd
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.mcp import MCPServerStreamableHttp
import random
import re
import hashlib
import numpy as np
import time

import json
from typing import Any

try:
    # many versions re-export these
    from agents import ToolCallItem, ToolCallOutputItem
except ImportError:
    from agents.items import ToolCallItem, ToolCallOutputItem

import json

def _pretty_for_file(x) -> str:
    """
    Pretty-print for a human-readable trace file:
    - If x is a JSON string, parse it first
    - Render with indent
    - Replace '\\n' -> real newlines for readability (not valid JSON anymore, but great for logs)
    """
    if isinstance(x, str):
        s = x
        # If it looks like JSON, try parsing so we don't show \" everywhere
        if s.lstrip().startswith(("{", "[")):
            try:
                obj = json.loads(s)
                return json.dumps(obj, indent=2, ensure_ascii=False).replace("\\n", "\n")
            except Exception:
                # fall through
                pass
        # Non-JSON string: just make escaped newlines readable
        return s.replace("\\n", "\n").replace('\\"', '"')

    # Dict/list/etc
    return json.dumps(x, indent=2, ensure_ascii=False).replace("\\n", "\n")


def _print_tool_output_readable(output, p):
    """
    If output looks like OpenAI-style {content:[{type:'text', text:'...'}]},
    print the inner text (parsed + multiline) instead of dumping the wrapper.
    """
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, list):
            # print each text block nicely
            printed_any = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    p(_pretty_for_file(part["text"]))
                    printed_any = True
            if printed_any:
                return
    # fallback
    p(_pretty_for_file(output))

def _jsonable(x: Any):
    # pydantic models in newer SDKs
    if hasattr(x, "model_dump"):
        return x.model_dump(exclude_none=True, exclude_unset=True)
    return x

def _pretty(x: Any, max_chars: int = 20000) -> str:
    try:
        s = json.dumps(_jsonable(x), indent=2, ensure_ascii=False)
    except Exception:
        s = repr(x)
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + f"\n...[truncated {len(s)-max_chars} chars]"
    return s

def log_tool_trace(result, p):
    p("### TOOL TRACE (args + outputs) ###")

    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = _jsonable(item.raw_item)

            name = getattr(item.raw_item, "name", None) or (raw.get("name") if isinstance(raw, dict) else None)
            args = getattr(item.raw_item, "arguments", None) or (raw.get("arguments") if isinstance(raw, dict) else None)

            p(f"\n--- TOOL CALL: {name} ---")
            if isinstance(args, str):
                # this will parse JSON strings and print them nicely
                p(_pretty_for_file(args))
            elif args is not None:
                p(_pretty_for_file(args))
            else:
                p(_pretty_for_file(raw))

            if isinstance(raw, dict) and "output" in raw:
                p("\n--- TOOL OUTPUT (inline) ---")
                _print_tool_output_readable(raw["output"], p)

        elif isinstance(item, ToolCallOutputItem):
            raw = _jsonable(item.raw_item)

            p(f"\n--- TOOL OUTPUT ---")
            if isinstance(raw, dict) and "output" in raw:
                _print_tool_output_readable(raw["output"], p)
            else:
                _print_tool_output_readable(raw, p)

def gen_prompt(cancer, variant, therapy):
    combination_text = f"gene variant: {variant}, cancer type: {cancer}, therapy: {therapy}"

    prompt = f"""
For this exact combination of {combination_text}, determine whether CIViC contains matching records for each evidence type + significance below, and whether those records support that significance.

Answer choices:
A. Supports
B. Does Not Support
C. No Evidence

CONTEXT-SENSITIVE EVIDENCE RULES (important):
- Treat the user's inputs (molecular profile, disease, therapy) as REQUIRED filters.
- Only count CIViC records that explicitly match ALL user-specified fields:
  - Disease must match exactly.
  - Therapy must match exactly as a set (order does not matter).
  - Exception: if Therapy Interaction Type is SUBSTITUTES, overlap is sufficient only for single-drug queries; for multi-drug queries the full combination must match.
- Molecular Profile matching:
  - If the QUERY MP contains "AND": only include evidence MPs that contain ALL components (no partial matches).
  - If the QUERY MP contains "OR": include evidence MPs if ANY component matches.
  - Otherwise (single MP): include evidence MPs that match that term; do NOT include evidence MPs with "AND".

TASK:
- For each item (<evidence type> — <significance>):
  - Return A if there exists ≥1 matching record with evidenceDirection = SUPPORTS.
  - Return B if there exists ≥1 matching record with evidenceDirection = DOES_NOT_SUPPORT.
  - Return A,B if both SUPPORTS and DOES_NOT_SUPPORT exist for that item.
  - If any matching record has evidenceDirection = NA / N/A / missing, treat direction as unknown and return A,B for that item.
  - Return C only if there are zero matching records for that evidence type + significance.
- The significance label "N/A" refers ONLY to CIViC records whose significance field is NA or N/A (not UNCERTAIN_SIGNIFICANCE).

Output exactly one line per item below (23 lines), in this exact format:
<evidence type> — <significance>: <A|B|A,B|C>

Items (use exactly this order, and exactly these labels):
predictive — sensitivity_response: <A|B|A,B|C>
predictive — resistance: <A|B|A,B|C>
predictive — adverse_response: <A|B|A,B|C>
predictive — reduced_sensitivity: <A|B|A,B|C>
predictive — N/A: <A|B|A,B|C>

prognostic — better_outcome: <A|B|A,B|C>
prognostic — poor_outcome: <A|B|A,B|C>
prognostic — N/A: <A|B|A,B|C>

diagnostic — positive: <A|B|A,B|C>
diagnostic — negative: <A|B|A,B|C>

predisposing — predisposition: <A|B|A,B|C>
predisposing — protectiveness: <A|B|A,B|C>
predisposing — uncertain_significance: <A|B|A,B|C>
predisposing — N/A: <A|B|A,B|C>

oncogenic — oncogenicity: <A|B|A,B|C>
oncogenic — protectiveness: <A|B|A,B|C>
oncogenic — N/A: <A|B|A,B|C>

functional — gain_of_function: <A|B|A,B|C>
functional — loss_of_function: <A|B|A,B|C>
functional — unaltered_function: <A|B|A,B|C>
functional — neomorphic: <A|B|A,B|C>
functional — dominant_negative: <A|B|A,B|C>
functional — unknown: <A|B|A,B|C>

FINAL OUTPUT RULES:
- Output ONLY the 23 lines.
- No prose, no markdown, no explanations.
"""
    return prompt


def sanitize_filename(name: str, replacement: str = "_", max_length: int = 255, ensure_unique: bool = False) -> str:
    name = str(name).replace("::", "-")
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    name = re.sub(invalid_chars, replacement, name)
    name = re.sub(re.escape(replacement) + r"{2,}", replacement, name)
    name = name.strip(replacement)

    if len(name) > max_length:
        if ensure_unique:
            hash_suffix = hashlib.sha1(name.encode()).hexdigest()[:8]
            trunc_length = max_length - len(hash_suffix) - 1
            name = name[:trunc_length].rstrip(replacement)
            name = f"{name}{replacement}{hash_suffix}"
        else:
            name = name[:max_length].rstrip(replacement)

    return name


class MCPToolFilter(logging.Filter):
    def filter(self, record):
        if record.levelname != "DEBUG":
            return False
        if not record.name.startswith("openai.agents"):
            return False
        msg = record.getMessage()
        return msg.startswith("Invoking MCP tool") or msg.startswith("MCP tool")


class NewlineFormatter(logging.Formatter):
    def format(self, record):
        s = super().format(record)
        return s.replace("\\n", "\n")


# --- Auth ---
with open("open_ai_key.txt", "r") as file:
    OPENAI_API_KEY = file.read().rstrip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Remote MCP config ---
REMOTE_MCP_URL = "https://civic-mcp-server.larscivic.workers.dev/mcp"
server_params = {
    "url": REMOTE_MCP_URL,
    "timeout": 120,
}


async def run_query_with_logging(agent: Agent, run_id: str, query: str) -> str:
    log_dir = r"data\one_shot_mcp"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        p = functools.partial(__builtins__.print, file=log_file, flush=True)

        # Per-run logging handler (restore after)
        root = logging.getLogger()
        old_handlers = root.handlers[:]
        old_level = root.level

        try:
            handler = logging.StreamHandler(log_file)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(NewlineFormatter("%(message)s"))
            handler.addFilter(MCPToolFilter())

            root.handlers = [handler]
            root.setLevel(logging.DEBUG)

            t1 = time.time()
            result = await Runner.run(agent, query)
            t2 = time.time()

            log_tool_trace(result, p)

            p("### LLM OUTPUT ###")
            p(result.final_output)
            p("### TIME (Seconds) ###")
            p(round(t2 - t1, 5))

        finally:
            root.handlers = old_handlers
            root.setLevel(old_level)

    return log_path


async def main():
    # --- Load + prep df (same logic as your script) ---
    df = pd.read_csv(r"data\CIViC_evidence_extracts_clinical_trials_curators.csv")

    df["therapies"] = df["therapies"].replace("", np.nan).fillna("None")
    df["disease_name"] = df["disease_name"].replace("", np.nan).fillna("None")

    df["entities"] = df["molecularProfile_name"] + "_" + df["disease_name"] + "_" + df["therapies"]
    df["evidence_summary"] = df["evidenceType"] + "_" + df["evidenceDirection"] + "_" + df["significance"]

    # Avoid NaNs breaking sanitize/apply
    df = df[df["entities"].notna()].copy()

    print(len(set(df["entities"])))

    labels = [
        "_predictive",
        "_diagnostic",
        "_prognostic",
        "_predisposing",
        "_oncogenic",
        "_functional",
        ".txt",
    ]

    pattern = r"(" + "|".join(map(re.escape, labels)) + r")"

    existing_dir = r"data\QA_eval_civic_mcp_evidence_type"
    output_dir = r"data\one_shot_mcp"
    os.makedirs(output_dir, exist_ok=True)

    existing_files = set(
        sanitize_filename(re.sub(pattern, "", fname), replacement="_")
        for fname in os.listdir(existing_dir)
        if fname.endswith(".txt")
    )

    new_output_files = set(
        sanitize_filename(fname.replace(".txt", ""), replacement="_")
        for fname in os.listdir(output_dir)
        if fname.endswith(".txt")
    )

    df["sanitized_entities"] = df["entities"].apply(lambda x: sanitize_filename(x, replacement="_"))

    df = df[df["sanitized_entities"].isin(existing_files)]
    df = df[~df["sanitized_entities"].isin(new_output_files)]

    print(len(set(df["entities"])))

    groupings = list(set(df["entities"]))

    # --- Create ONE OpenAI client, ONE MCP connection, ONE Agent ---
    openai_client = AsyncOpenAI()

    srv = MCPServerStreamableHttp(
        name="CIViC tools",
        params=server_params,
        cache_tools_list=True,
        max_retry_attempts=3,
    )

    async with srv:
        agent = Agent(
            name="CIViC-Assistant",
            instructions=(
                "You are an expert biomedical annotator trained to assess variant evidence from the "
                "Clinical Interpretations of Variants in Cancer (CIViC) knowledgebase given a specific "
                "gene variant + cancer + therapy.\n\n"
                "Use the tools to answer oncology questions for the CIViC knowledgebase."
            ),
            model=OpenAIChatCompletionsModel(
                model="gpt-5-2025-08-07",
                openai_client=openai_client,
            ),
            model_settings=ModelSettings(temperature=1),
            mcp_servers=[srv],
        )

        x = 0
        for idx, grouping in enumerate(groupings):
            print(idx, sanitize_filename(grouping, replacement="_"))

            if len(grouping.split("_")) > 3 or len(grouping) > 100:
                print("skipped")
                continue

            variant, cancer, therapy = grouping.split("_")
            print(variant)
            user_prompt = gen_prompt(cancer, variant, therapy)

            await run_query_with_logging(
                agent=agent,
                run_id=sanitize_filename(grouping, replacement="_"),
                query=user_prompt,
            )

            await asyncio.sleep(30 + random.uniform(0, 2))


if __name__ == "__main__":
    asyncio.run(main())