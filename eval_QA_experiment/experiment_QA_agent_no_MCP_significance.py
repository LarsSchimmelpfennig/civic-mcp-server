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
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

significance_types_map = {
    'predictive': ['sensitivity_response', 'reduced_sensitivity', 'resistance', 'adverse_response'],
    'diagnostic': ['positive', 'negative'],
    'prognostic': ['better_outcome', 'poor_outcome'],
    'predisposing': ['predisposition', 'protectiveness'],
    'oncogenic': ['oncogenicity', 'protectiveness'],
    'functional': ['gain_of_function', 'loss_of_function', 'unaltered_function', 'neomorphic', 'dominant_negative', 'uncertain_significance']
}

def read_llm_answer_letter(text: str) -> str:
    """
    Extract the answer letter(s) directly from file content.
    """
    return text.strip().upper()

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

def gen_prompt(gene_variant, cancer_type, therapy, evidence_type, significance):

    messages = [
        {
            "role": "system",
            "content": (
                """You are an expert biomedical annotator trained to assess variant evidence from the CIViC knowledgebase given a specific gene variant + cancer + therapy.\n\n"""

                "**Definition of Evidence Types**\n"
                "• Diagnostic – Evidence pertains to a variant’s impact on patient diagnosis (cancer subtype).\n"
                "• Predictive – Evidence pertains to a variant’s effect on therapeutic response.\n"
                "• Prognostic – Evidence pertains to a variant’s impact on disease progression, severity, or patient survival.\n"
                "• Predisposing – Evidence pertains to a germline molecular profile’s role in conferring susceptibility to disease (including pathogenicity evaluations).\n"
                "• Oncogenic – Evidence pertains to a somatic variant’s involvement in tumor pathogenesis as described by the Hallmarks of Cancer.\n"
                "• Functional – Evidence pertains to a variant that alters biological function from the reference state.\n\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Does the {evidence_type} evidence for this combination of gene variant: {gene_variant}, cancer type: {cancer_type}, and therapy: {therapy} "
                f"support the significance: {significance} in CIViC?\n"
                "A. Supports\n"
                "B. Does Not Support\n"
                "C. No Evidence\n"
                f"Choose all that apply. If both supporting and not supporting evidence exist for the significance {significance}, return A,B. "
                "Return C if there is no evidence with this significance. Return B only if there is evidence with this significance"
            )
        }
    ]

    return messages

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

with open('open_ai_key.txt', 'r') as file:
    OPENAI_API_KEY = file.read().rstrip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------- CONFIG --------------------
CSV_PATH = "data/CIViC_evidence_extracts_clinical_trials_curators.csv"
TXT_DIR  = "data/QA_eval_civic_no_mcp_evidence_type"

OUT_DIR = "data/QA_eval_civic_no_mcp_significance"


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
    sanitize_filename(fname.replace('.txt', '').rsplit('_', 1)[0], replacement='_')
    for fname in os.listdir(TXT_DIR)
    if fname.endswith('.txt')
)

# 2. Sanitize your DataFrame entities the same way
df['sanitized_entities'] = df['entities'].apply(lambda x: sanitize_filename(x, replacement='_'))

df = df[df['sanitized_entities'].isin(existing_files)]

print(len(set(df['entities'])))

evidence_types = ['predictive', 'diagnostic', 'prognostic', 'predisposing', 'oncogenic', 'functional']

df_etypes_by_entity = (
df.assign(
        evidenceType_lower=df["evidenceType"].astype(str).str.strip().str.lower()
    )
    .groupby("sanitized_entities")["evidenceType_lower"]
    .agg(lambda s: set(t for t in s if t))  # unique, non-empty
    .to_dict()
)

entity_etype_map = etype_mapping(TXT_DIR)

# # Intersect entity_etype_map with what exists in df for that entity
# filtered_entity_etype_map = {}
# removed_entities = 0
# removed_types = 0

# for ent_key, etypes_from_files in entity_etype_map.items():
#     # ensure list is lowercase/clean
#     etypes_from_files = [str(e).strip().lower() for e in etypes_from_files if e]
#     print(ent_key)
#     print(etypes_from_files)

#     allowed = df_etypes_by_entity.get(ent_key, set())
#     print(allowed)
#     keep = [e for e in etypes_from_files if e in allowed]

#     removed_types += (len(etypes_from_files) - len(keep))

#     # Option A (recommended): drop entities with no matching types
#     if keep:
#         filtered_entity_etype_map[ent_key] = keep
#     else:
#         removed_entities += 1
#         #print(ent_key)

# print(f"Entities kept: {len(filtered_entity_etype_map)}")
# print(f"Entities removed (no matching etypes in df): {removed_entities}")
# print(f"Evidence types removed by filtering: {removed_types}")

# sys.exit()


for entity in df['entities']:
    entity_sanitized = sanitize_filename(entity, replacement='_')
    etypes = entity_etype_map[entity_sanitized]
    print(entity, etypes)
    for etype in etypes:
        significances = significance_types_map[etype]
        #print(significances)
        for significance in significances:
            
            file_name = entity_sanitized+'__'+etype+'_'+significance+'.txt'

            #remove entities that already appear in the significance output_dir

            OUT_DIR = "data/QA_eval_civic_no_mcp_significance"

            if file_name in os.listdir(OUT_DIR):
                continue

            print(file_name)

            gene_variant, cancer_type, therapy = entity.split('_')

            prompt = gen_prompt(gene_variant, cancer_type, therapy, etype, significance)

            api_key = os.getenv("OPENAI_API_KEY")
            # Initialize client (replace with your own API key)
            client = openai.OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                        model="gpt-5-2025-08-07",                    
                        messages=prompt,
                        temperature=1
                    )

            msg = response.choices[0].message.content
            time.sleep(10)
            #print(msg)

            with open(os.path.join(OUT_DIR, file_name), 'w') as file:
                file.write(msg)


    
