import asyncio, os, sys, functools, logging
import pandas as pd
from datetime import datetime
from openai import AsyncOpenAI
import random, re, hashlib
import numpy as np
import time

SYSTEM_PROMPT = """You are an expert biomedical annotator trained to assess variant evidence from the CIViC knowledgebase given a specific gene variant + cancer + therapy.\n\n"""

MODEL_NAME = "gpt-5-2025-08-07"
TEMPERATURE = 1

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
    """
    Sanitize a string to be used as a safe filename.

    Args:
        name (str): Original filename (without path or extension).
        replacement (str): Replacement for unsafe characters (default: "_").
        max_length (int): Max length for filename (default: 255).
        ensure_unique (bool): Whether to append hash to ensure uniqueness when truncated.

    Returns:
        str: Safe, valid filename.
    """
    name = name.replace('::', '-')
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    name = re.sub(invalid_chars, replacement, name)
    name = re.sub(re.escape(replacement) + r'{2,}', replacement, name)
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

class NewlineFormatter(logging.Formatter):
    def format(self, record):
        s = super().format(record)
        return s.replace("\\n", "\n")

def load_openai_key(path: str = "open_ai_key.txt") -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().rstrip()

async def run_query_with_logging(run_id: str, query: str, client: AsyncOpenAI) -> str:
    log_dir = os.path.join("data", "one_shot_no_mcp")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        print_to_log = functools.partial(__builtins__.print, file=log_file, flush=True)

        handler = logging.StreamHandler(log_file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(NewlineFormatter("%(message)s"))
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        t1 = time.time()
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
        )
        t2 = time.time()

        msg = response.choices[0].message.content or ""
        print_to_log("### LLM OUTPUT ###")
        print_to_log(msg)
        print_to_log("### TIME (Seconds) ###")
        print_to_log(round(t2 - t1, 5))

    return log_path


async def main():
    OPENAI_API_KEY = load_openai_key()
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    try:
        df = pd.read_csv(os.path.join("data", "CIViC_evidence_extracts_clinical_trials_curators.csv"))

        df["therapies"] = df["therapies"].replace("", np.nan).fillna("None")
        df["disease_name"] = df["disease_name"].replace("", np.nan).fillna("None")

        df["entities"] = df["molecularProfile_name"] + "_" + df["disease_name"] + "_" + df["therapies"]
        df["sanitized_entities"] = df["entities"].apply(lambda x: sanitize_filename(x, replacement="_"))

        labels = ["_predictive", "_diagnostic", "_prognostic", "_predisposing", "_oncogenic", "_functional", ".txt"]
        pattern = r"(" + "|".join(map(re.escape, labels)) + r")"

        existing_dir = os.path.join("data", "QA_eval_civic_mcp_evidence_type")
        existing_files = set(
            sanitize_filename(re.sub(pattern, "", fname), replacement="_")
            for fname in os.listdir(existing_dir)
            if fname.endswith(".txt")
        )

        new_dir = os.path.join("data", "one_shot_no_mcp")
        os.makedirs(new_dir, exist_ok=True)
        new_output_files = set(
            sanitize_filename(fname.replace(".txt", ""), replacement="_")
            for fname in os.listdir(new_dir)
            if fname.endswith(".txt")
        )

        df = df[df["sanitized_entities"].isin(existing_files)]
        df = df[~df["sanitized_entities"].isin(new_output_files)]

        x = 0
        for idx, grouping in enumerate(list(set(df["entities"]))):
            print(idx, sanitize_filename(grouping, replacement="_"))

            if len(grouping.split("_")) > 3 or len(grouping) > 100:
                print("skipped")
                continue

            variant, cancer, therapy = grouping.split("_")
            user_prompt = gen_prompt(cancer, variant, therapy)

            run_id = sanitize_filename(grouping, replacement="_")
            await run_query_with_logging(run_id, user_prompt, client)

            await asyncio.sleep(30)   # IMPORTANT: async sleep inside async main
    finally:
        await client.close()          # important cleanup :contentReference[oaicite:2]{index=2}


if __name__ == "__main__":
    asyncio.run(main())