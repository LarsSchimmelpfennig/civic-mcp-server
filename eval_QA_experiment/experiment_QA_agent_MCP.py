import asyncio, os, sys, functools, logging
import pandas as pd
from datetime import datetime
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.mcp import MCPServerStdio
from io import StringIO
import csv
import random, re, hashlib
import numpy as np
import time

def sanitize_filename(name: str, replacement: str = "_", max_length: int = 255, ensure_unique: bool = False) -> str:
    """
    Sanitize a string to be used as a safe filename.

    Args:
        name (str): Original filename (without path or extension).
        replacement (str): Replacement for unsafe characters (default: "-").
        max_length (int): Max length for filename (default: 255).
        ensure_unique (bool): Whether to append hash to ensure uniqueness when truncated.

    Returns:
        str: Safe, valid filename.
    """

    name = name.replace('::', '-')
    # Invalid characters for Windows and general use
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    name = re.sub(invalid_chars, replacement, name)

    # Collapse multiple replacement chars
    name = re.sub(re.escape(replacement) + r'{2,}', replacement, name)

    # Strip leading/trailing replacement characters
    name = name.strip(replacement)

    # Truncate and add hash if needed
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
        # only look at the openai.agents loggers
        if not record.name.startswith("openai.agents"):
            return False

        msg = record.getMessage()
        # let through both invocation _and_ return messages
        return (
            msg.startswith("Invoking MCP tool")
            or msg.startswith("MCP tool")
        )

class NewlineFormatter(logging.Formatter):
    def format(self, record):
        # first let the base Formatter build the string
        s = super().format(record)
        # then un‐escape any “\n” sequences into real newlines
        return s.replace("\\n", "\n")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Set up the MCP server params (only once) ===
server_params = {
    "command": "fastmcp",
    "args": ["run", "server_MCP_CIViC.py:mcp", "--transport", "stdio"],
    "env": {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "MCP_DEBUG": "1",
    },
    "errlog": sys.stderr,
}

# === Main function to run a single query and log it ===
async def run_query_with_logging(run_id, query):
    log_dir = "data\QA_eval_civic_mcp_evidence_type"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.txt")

    # Open log file and redirect print/logging to it
    with open(log_path, "w", encoding="utf-8") as log_file:

        # Redirect print
        print = functools.partial(__builtins__.print, file=log_file, flush=True)

        # Set up logging
        handler = logging.StreamHandler(log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(NewlineFormatter("%(message)s"))
        handler.addFilter(MCPToolFilter())
        logging.getLogger().handlers = [handler]
        logging.getLogger().setLevel(logging.DEBUG)

        # Start MCP server
        srv = MCPServerStdio(name="CIViC tools", params=server_params)
        await srv.__aenter__()

        agent = Agent(
            name="CIViC-Assistant",
            instructions = (
                """You are an expert biomedical annotator trained to assess variant evidence from the Clinical Interpretations of Variants in Cancer (CIViC) knowledgebase given a specific gene variant + cancer + therapy.\n\n"""
                "Use the tools to answer oncology questions for the CIViC knowledgebase."
                ),            
            model=OpenAIChatCompletionsModel(
                model="gpt-5-2025-08-07",
                openai_client=AsyncOpenAI()
            ),
            model_settings=ModelSettings(temperature=1),
            mcp_servers=[srv],
        )

        result = await Runner.run(agent, query)

        print("### LLM OUTPUT ###")
        print(result.final_output)

        await srv.__aexit__(None, None, None)

    # Return path for run_index.csv logging
    return log_path

# === Entry point ===
if __name__ == "__main__":
    random.seed(1)

    df = pd.read_csv('data\CIViC_evidence_extracts_clinical_trials_curators.csv')

    df['therapies'] = (
    df['therapies']
      .replace('', np.nan)      # if empty strings appear
      .fillna('None')           # replace NaN → 'None'
    )

    df['disease_name'] = (
    df['disease_name']
      .replace('', np.nan)      # if empty strings appear
      .fillna('None')           # replace NaN → 'None'
    )

    #df = df[df['status'] == 'ACCEPTED']
    print(len(df))
    #df = df[df['evidenceType'] != 'PREDICTIVE']
    print(len(df))

    #df = df[(df['status'] == 'ACCEPTED') & (df['evidenceDirection'] == 'SUPPORTS')]
    df['entities'] = df['molecularProfile_name'] + '_' + df['disease_name'] + '_' + df['therapies']
    df['evidence_summary'] = df['evidenceType'] + '_' + df['evidenceDirection'] + '_' + df['significance']

    df["entities"].dropna()
    
    print(len(set(df['entities'])))

    existing_files = set(
        sanitize_filename(fname.replace('.txt', ''), replacement='_')
        for fname in os.listdir('data/QA_eval_civic_mcp_evidence_type')
        if fname.endswith('.txt')
    )

    # 2. Sanitize your DataFrame entities the same way
    df['sanitized_entities'] = df['entities'].apply(lambda x: sanitize_filename(x, replacement='_'))

    # 3. Filter out rows with already processed entities
    df = df[~df['sanitized_entities'].isin(existing_files)]

    #include = ['FUS-ERG Fusion_Acute Myeloid Leukaemia With FUS-ERG Fusion_None', 'TPM3-NTRK1 Fusion_Lung Carcinoma_Larotrectinib']
    #Had previously included this 'BRAF V600E_Colorectal Cancer_None'

    #df = df[df['sanitized_entities'].isin(include)]

    print(len(set(df['entities'])))

    #print("Eval entities:", eval_entities)

    #df = df.head(5)
    #print(len(df))

    random.seed(1)
    groupings = random.sample(list(set(df['entities'])), k=1)
    print(len(groupings))
    print(groupings)
    sys.exit()

    evidence_types = ['predictive', 'diagnostic', 'prognostic', 'predisposing', 'oncogenic', 'functional']

    #sys.exit()
    
 
    for idx, grouping in enumerate(groupings):
        print(idx, sanitize_filename(grouping, replacement='_'))
        for evidence_type in evidence_types:
        
            if len(grouping.split('_')) > 3 or len(grouping) > 100:
                print('skipped')
                continue

            gene_variant, cancer_type, therapy = grouping.split('_')

            user_prompt = (
                f"Does the CIViC evidence for this combination of gene variant: {gene_variant}, cancer type: {cancer_type}, and therapy: {therapy} pertain to the "
                f"{evidence_type} evidence type?\n"
                "A. Yes\n"
                "B. No\n"
                "C. Unsure\n"
                "Answer with only the letter corresponding to your choice."
            )

            log_path = asyncio.run(run_query_with_logging(sanitize_filename(grouping, replacement='_')+f'_{evidence_type}', user_prompt))
            time.sleep(10)


    
