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

with open('open_ai_key.txt', 'r') as file:
    OPENAI_API_KEY = file.read().rstrip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
    log_dir = "data/QA_eval_civic_mcp_significance"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}")

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



def read_llm_answer_letter(text: str) -> str:
    """Extract the single-letter answer ('A','B','C') from file content."""
    lines = [ln.strip() for ln in text.splitlines()]
    for i, ln in enumerate(lines):
        if ln.upper().startswith("### LLM OUTPUT ###"):
            for j in range(i + 1, len(lines)):
                cand = lines[j].strip()
                if cand:
                    return cand[:1].upper()
            break
    for ln in reversed(lines):
        if ln:
            return ln[:1].upper()
    return ""

#create a mapping from TXT_DIR files to supported etypes
#TODO Make sure that the eval file allowed multipe evidence types per entitiy.
def etype_mapping(TXT_DIR):

    entity_etype_map = {}

    for fname in os.listdir(TXT_DIR):
        if not fname.endswith(".txt"):
            continue

        base = fname[:-4]  # strip .txt

        # base looks like: <SANITIZED_ENTITY>_<evidence_type>
        entity_part, etype_part = base.rsplit("_", 1)
        etype_part = etype_part.lower()

        # Ensure the entity key is sanitized (idempotent)
        entity_key = sanitize_filename(entity_part, replacement="_")

        if entity_key not in entity_etype_map:
            entity_etype_map[entity_key] = []

        with open(os.path.join(TXT_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        letter = read_llm_answer_letter(content)

        if letter == "A":
            entity_etype_map[entity_key].append(etype_part)

    return entity_etype_map

# === Entry point ===
if __name__ == "__main__":

    #iterate through each mcp_evidence_type file to get the list of entities
    #get the list of unique significances that appear associated with those entities
    #Ask the Support, Does not support, no evidence question where It can answer A and B

    #If it did not identify the correct evidence type then it answers no evidence automatically.

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

    print(len(set(df['entities'])))

    #Get the list of entities from the evidence type output_dir

    # Existing txt files (entity sanitized + "_<evidence_type>.txt")
    TXT_DIR = "data/QA_eval_civic_mcp_evidence_type"
    evidence_type_files = set(
        sanitize_filename(fname.replace('.txt', '').rsplit('_', 1)[0], replacement='_')
        for fname in os.listdir(TXT_DIR)
        if fname.endswith('.txt')
    )

    df = df[df['sanitized_entities'].isin(evidence_type_files)]

    # fname will be like 'triplet_etype*significance.txt'

    #Find what evidence types were supported with 'A'

    entity_etype_map = etype_mapping(TXT_DIR)
    print(len(entity_etype_map))
    #print(entity_etype_map)

    #for each supported evidence type ask about each significance

    types_to_csv = {
        "predictive;sensitivity_response": ("PREDICTIVE", "SENSITIVITYRESPONSE"),
        "predictive;reduced_sensitivity":  ("PREDICTIVE", "REDUCED_SENSITIVITY"),
        "predictive;resistance":           ("PREDICTIVE", "RESISTANCE"),
        "predictive;adverse_response":     ("PREDICTIVE", "ADVERSE_RESPONSE"),
        "diagnostic;positive":             ("DIAGNOSTIC", "POSITIVE"),
        "diagnostic;negative":             ("DIAGNOSTIC", "NEGATIVE"),
        "prognostic;better_outcome":       ("PROGNOSTIC", "BETTER_OUTCOME"),
        "prognostic;poor_outcome":         ("PROGNOSTIC", "POOR_OUTCOME"),
        "predisposing;predisposition":     ("PREDISPOSING", "PREDISPOSITION"), #make sure these are the only ones counted.
        "predisposing;protectiveness":     ("PREDISPOSING", "PROTECTIVENESS"),
        "oncogenic;oncogenicity":          ("ONCOGENIC",  "ONCOGENICITY"),
        "oncogenic;protectiveness":        ("ONCOGENIC",  "PROTECTIVENESS"),
        "functional;gain_of_function":     ("FUNCTIONAL", "GAIN_OF_FUNCTION"),
        "functional;loss_of_function":     ("FUNCTIONAL", "LOSS_OF_FUNCTION"),
        "functional;unaltered_function":   ("FUNCTIONAL", "UNALTERED_FUNCTION"),
        "functional;neomorphic":           ("FUNCTIONAL", "NEOMORPHIC"),
        "functional;dominant_negative":    ("FUNCTIONAL", "DOMINANT_NEGATIVE"),
        "functional;uncertain_significance": ("FUNCTIONAL", "UNCERTAIN_SIGNIFICANCE"),
    }

    evidence_types = ['predictive', 'diagnostic', 'prognostic', 'predisposing', 'oncogenic', 'functional']
    significance_types_map = {
        'predictive': ['sensitivity_response', 'reduced_sensitivity', 'resistance', 'adverse_response'],
        'diagnostic': ['positive', 'negative'],
        'prognostic': ['better_outcome', 'poor_outcome'],
        'predisposing': ['predisposition', 'protectiveness'],
        'oncogenic': ['oncogenicity', 'protectiveness'],
        'functional': ['gain_of_function', 'loss_of_function', 'unaltered_function', 'neomorphic', 'dominant_negative', 'uncertain_significance']
    }

    #Only ask about the significanes for etypes that do exist
    #Return 'A/B' if there is conflicting evidence.

    #find etyoes that match the expected

    df_etypes_by_entity = (
    df.assign(
            evidenceType_lower=df["evidenceType"].astype(str).str.strip().str.lower()
        )
        .groupby("sanitized_entities")["evidenceType_lower"]
        .agg(lambda s: set(t for t in s if t))  # unique, non-empty
        .to_dict()
    )

    # Intersect entity_etype_map with what exists in df for that entity
    filtered_entity_etype_map = {}
    removed_entities = 0
    removed_types = 0

    for ent_key, etypes_from_files in entity_etype_map.items():
        # ensure list is lowercase/clean
        etypes_from_files = [str(e).strip().lower() for e in etypes_from_files if e]

        allowed = df_etypes_by_entity.get(ent_key, set())
        keep = [e for e in etypes_from_files if e in allowed]

        removed_types += (len(etypes_from_files) - len(keep))

        # Option A (recommended): drop entities with no matching types
        if keep:
            filtered_entity_etype_map[ent_key] = keep
        else:
            removed_entities += 1
            print(ent_key)

    print(f"Entities kept: {len(filtered_entity_etype_map)}")
    print(f"Entities removed (no matching etypes in df): {removed_entities}")
    print(f"Evidence types removed by filtering: {removed_types}")

    #sys.exit()

    print(len(df))
    df = df[df['sanitized_entities'].isin(list(entity_etype_map.keys()))]
    print(len(df))

    print(len(set(df['entities'])))

    i = 0
    for entity in df['entities']:
        entity_sanitized = sanitize_filename(entity, replacement='_')
        etypes = entity_etype_map[entity_sanitized]
        for etype in etypes:
            significances = significance_types_map[etype]
            #print(significances)
            for significance in significances:
                
                file_name = entity_sanitized+'__'+etype+'_'+significance+'.txt'

                #remove entities that already appear in the significance output_dir

                OUT_DIR = "data/QA_eval_civic_mcp_significance"

                if file_name in os.listdir(OUT_DIR):
                    continue

                print(file_name)
                i+=1
                print(i)

                gene_variant, cancer_type, therapy = entity.split('_')

                user_prompt = (
                    f"Does the {etype} evidence for this combination of gene variant: {gene_variant}, cancer type: {cancer_type}, and therapy: {therapy} "
                    f"support the significance: {significance} in CIViC?\n"
                    "A. Supports\n"
                    "B. Does Not Support\n"
                    "C. No Evidence\n"
                    f"Choose all that apply. If both supporting and not supporting evidence exist for the significance {significance}, return A,B. "
                    "Return C if there is no evidence with this significance. Return B only if there is evidence with this significance"
                )

                log_path = asyncio.run(run_query_with_logging(file_name, user_prompt))
                time.sleep(10)


    
