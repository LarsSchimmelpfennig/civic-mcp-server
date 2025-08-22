import os, re, json, hashlib, sys
import numpy as np
import pandas as pd
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def gen_prompt(gene_variant, cancer_type, therapy, evidence_type):

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
                f"Does the CIViC evidence for this combination of gene variant: {gene_variant}, cancer type: {cancer_type}, and therapy: {therapy} pertain to the "
                f"{evidence_type} evidence type?\n"
                "A. Yes\n"
                "B. No\n"
                "C. Unsure\n"
                "Answer with only the letter corresponding to your choice."
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

# -------------------- CONFIG --------------------
CSV_PATH = "data/CIViC_evidence_extracts_clinical_trials_curators.csv"
TXT_DIR  = "data/QA_eval_civic_mcp_evidence_type"

#convert file names back to the original groupings
#did before in eval

df = pd.read_csv(CSV_PATH)

df['therapies'] = df['therapies'].replace('', np.nan).fillna('None')
df['disease_name'] = df['disease_name'].replace('', np.nan).fillna('None')

#df = df[df['status'] == 'ACCEPTED']
#print(len(df))
#df = df[df['evidenceType'] != 'PREDICTIVE']

# df = df[(df['evidenceDirection'] == 'SUPPORTS')]

df['entities'] = (
    df['molecularProfile_name'] + '_' +
    df['disease_name'] + '_' +
    df['therapies']
)


existing_files = set(
    sanitize_filename(fname.replace('.txt', '').rsplit('_', 1)[0], replacement='_')
    for fname in os.listdir(TXT_DIR)
    if fname.endswith('.txt')
)

created_eval_files = set(
    sanitize_filename(fname.replace('.txt', '').rsplit('_', 1)[0], replacement='_')
    for fname in os.listdir('data/QA_eval_civic_no_mcp_evidence_type')
    if fname.endswith('.txt')
)

# 2. Sanitize DataFrame entities the same way
df['sanitized_entities'] = df['entities'].apply(lambda x: sanitize_filename(x, replacement='_'))

# Keep only the entities that have been processed
df = df[df['sanitized_entities'].isin(existing_files)]

df = df[~df['sanitized_entities'].isin(created_eval_files)]

entities = df['entities'].dropna().unique()

OUT_DIR = 'data\QA_eval_civic_no_mcp_evidence_type'

print(len(entities))
print(entities)
sys.exit()

i = 0
evidence_types = ['predictive', 'diagnostic', 'prognostic', 'predisposing', 'oncogenic', 'functional']
for entity in entities:
    i+=1
    print(i, entity, len(entities))
    for evidence_type in evidence_types:
        
        MP, disease, therapy = entity.split('_')

        prompt = gen_prompt(MP, disease, therapy, evidence_type)

        api_key = os.getenv("OPENAI_API_KEY")
        # Initialize client (replace with your own API key)
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
                    model="gpt-5-2025-08-07",                    
                    messages=prompt,
                    temperature=1
                )

        msg = response.choices[0].message.content
        #print(msg)

        with open(os.path.join(OUT_DIR, sanitize_filename(entity, replacement='_')+f'_{evidence_type}.txt'), 'w') as file:
            file.write(msg)