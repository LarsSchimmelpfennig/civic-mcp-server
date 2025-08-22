import os, re, json, hashlib, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score
)
from collections import defaultdict

types_to_csv = {
    "predictive_sensitivity_response": ("PREDICTIVE", "SENSITIVITYRESPONSE"),
    "predictive_reduced_sensitivity":  ("PREDICTIVE", "REDUCED_SENSITIVITY"),
    "predictive_resistance":           ("PREDICTIVE", "RESISTANCE"),
    "predictive_adverse_response":     ("PREDICTIVE", "ADVERSE_RESPONSE"),
    "diagnostic_positive":             ("DIAGNOSTIC", "POSITIVE"),
    "diagnostic_negative":             ("DIAGNOSTIC", "NEGATIVE"),
    "prognostic_better_outcome":       ("PROGNOSTIC", "BETTER_OUTCOME"),
    "prognostic_poor_outcome":         ("PROGNOSTIC", "POOR_OUTCOME"),
    "predisposing_predisposition":     ("PREDISPOSING", "PREDISPOSITION"), #make sure these are the only ones counted.
    "predisposing_protectiveness":     ("PREDISPOSING", "PROTECTIVENESS"),
    "oncogenic_oncogenicity":          ("ONCOGENIC",  "ONCOGENICITY"),
    "oncogenic_protectiveness":        ("ONCOGENIC",  "PROTECTIVENESS"),
    "functional_gain_of_function":     ("FUNCTIONAL", "GAIN_OF_FUNCTION"),
    "functional_loss_of_function":     ("FUNCTIONAL", "LOSS_OF_FUNCTION"),
    "functional_unaltered_function":   ("FUNCTIONAL", "UNALTERED_FUNCTION"),
    "functional_neomorphic":           ("FUNCTIONAL", "NEOMORPHIC"),
    "functional_dominant_negative":    ("FUNCTIONAL", "DOMINANT_NEGATIVE"),
    "functional_uncertain_significance": ("FUNCTIONAL", "UNCERTAIN_SIGNIFICANCE"),
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

def sanitize_filename(name: str, replacement: str = "_", max_length: int = 255, ensure_unique: bool = False) -> str:
    """
    Sanitize a string to be used as a safe filename.
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


# -------------------- CONFIG --------------------
CSV_PATH = "data/CIViC_evidence_extracts_clinical_trials_curators.csv"
#TXT_DIR  = "data/QA_eval_civic_mcp_significance"
TXT_DIR  = "data/QA_eval_civic_no_mcp_significance"


# -------------------- LOAD & FILTER CSV --------------------
df = pd.read_csv(CSV_PATH)

df['therapies'] = df['therapies'].replace('', np.nan).fillna('None')
df['disease_name'] = df['disease_name'].replace('', np.nan).fillna('None')
#df = df[(df['evidenceDirection'] == 'SUPPORTS')]

df['entities'] = (
    df['molecularProfile_name'] + '_' +
    df['disease_name'] + '_' +
    df['therapies']
)

df['evidence_summary'] = df['evidenceType'] + '_' + df['evidenceDirection'] + '_' + df['significance']

# Existing txt files (entity sanitized + "_<evidence_type>.txt")
existing_files = set(
    sanitize_filename(fname.replace('.txt', '').rsplit('_', 1)[0], replacement='_')
    for fname in os.listdir('data/QA_eval_civic_mcp_evidence_type')
    if fname.endswith('.txt')
)

# Sanitize entities the same way and keep only those present in TXT_DIR
df['sanitized_entities'] = df['entities'].apply(lambda x: sanitize_filename(x, replacement='_'))
df = df[df['sanitized_entities'].isin(existing_files)]

# These are the *index labels* we'll use everywhere (sanitized keys)
index_labels = pd.Index(sorted(df['sanitized_entities'].dropna().unique()), name='sanitized_entity')
print(len(index_labels), len(df))


# -------------------- BUILD INDICATOR MATRICES --------------------
# IMPORTANT: index by sanitized names so this aligns with what's in TXT_DIR
y_pred = pd.DataFrame(0, index=index_labels, columns=types_to_csv.keys(), dtype=int)

# (You likely fill y_true later from df; leave it indexed by sanitized names.)

def read_llm_answer_letter(text: str) -> str:
    """
    Extract an answer from the region after '### LLM OUTPUT ###'.
    Returns one of: 'A', 'B', 'C', or 'A,B'. Prefers the bottom-most non-empty line.
    Also supports JSON like: {"answer":"A,B"}.
    Special case: 'C. No Evidence' maps to 'C'.
    """
    TAG = "### LLM OUTPUT ###"
    lines = text.splitlines()

    # Find start region (line after the tag), else the whole file
    start = -1
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith(TAG):
            start = i
            break
    block = [ln.strip() for ln in (lines[start+1:] if start >= 0 else lines) if ln.strip()]
    if not block:
        return ""

    # Prefer bottom-most non-empty line
    last = block[-1]

    # Explicitly catch "No Evidence"
    if re.search(r"\bNO\s+EVIDENCE\b", last, flags=re.I):
        return "C"

    # 1) Try JSON anywhere in the block
    joined = "\n".join(block)
    m = re.search(r'"answer"\s*:\s*"([^"]+)"', joined, flags=re.I)
    raw = m.group(1) if m else last

    # 2) Normalize/canonicalize
    raw = raw.upper()
    # keep only A/B/C and commas (handles "Answer: A,B" etc.)
    cleaned = re.sub(r'[^ABC,]', '', raw)

    # Canonicalize order & duplicates
    letters = [c for c in cleaned.split(',') if c in {'A','B','C'}]
    order = {'A':0, 'B':1, 'C':2}
    uniq_sorted = []
    for x in sorted(dict.fromkeys(letters), key=lambda k: order[k]):
        uniq_sorted.append(x)

    # Map to allowed outputs
    if not uniq_sorted:
        return ""
    if uniq_sorted == ['A','B']:
        return 'A,B'
    if uniq_sorted == ['A']:
        return 'A'
    if uniq_sorted == ['B']:
        return 'B'
    if uniq_sorted == ['C']:
        return 'C'

    # If something odd like 'A,C' slipped through, pick a sane fallback:
    if 'C' in uniq_sorted:
        return 'C'
    return uniq_sorted[0]

used_files = 0
skipped_files = 0

for fname in os.listdir(TXT_DIR):
    if not fname.endswith(".txt"):
        continue

    base = fname[:-4]  # strip .txt
    if "_" not in base:
        skipped_files += 1
        continue

    # base looks like: <SANITIZED_ENTITY>_<evidence_type>
    entity_part, significance_part = base.rsplit("__", 1)
    significance_part = significance_part.lower()

    if entity_part not in y_pred.index:
        # Not in our filtered set; skip but show what we saw
        print(f"[skip] missing index for: '{entity_part}' -> key '{entity_part}'")
        skipped_files += 1
        continue

    with open(os.path.join(TXT_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    letter = read_llm_answer_letter(content)

    #print(entity_part, significance_part, letter)

    if letter == "A":
        y_pred.at[entity_part, significance_part] = 1
    elif letter == "B":
        y_pred.at[entity_part, significance_part] = 2
    elif letter == "A,B":
        y_pred.at[entity_part, significance_part] = 3
    elif letter == "C":
        y_pred.at[entity_part, significance_part] = 0
    else:
        # No recognized letter; treat as skip
        skipped_files += 1
        print('no letter', fname)
        continue

    used_files += 1

print(f"Parsed files: {used_files}, Skipped files: {skipped_files}")
#print(y_pred)

df['etype'] = df['evidenceType'].astype(str).str.strip().str.lower()

#df[['entities', 'evidenceType', 'significance']].to_csv('data/eval_inspect_entities.csv')

#print(y_true)

# Build reverse map from CSV tuple -> underscore key
csv_to_types = {v: k for k, v in types_to_csv.items()}  # e.g. ("PREDICTIVE","SENSITIVITYRESPONSE") -> "predictive_sensitivity_response"

all_labels = list(types_to_csv.keys())
y_true = pd.DataFrame(
    0,
    index=index_labels,                 # align to your sanitized index set
    columns=all_labels,
    dtype=int
)

for _, row in df.iterrows():
    entity = row['sanitized_entities']
    if entity not in y_true.index:
        continue

    # Normalize to the CSV-style tuple used in types_to_csv values
    etype_csv = str(row['evidenceType']).strip().upper()          # e.g. "PREDICTIVE"
    sig_csv   = str(row['significance']).strip().upper()          # e.g. "SENSITIVITYRESPONSE", "POSITIVE"
    # If your CSV sometimes has spaces/dashes/extra underscores, normalize them:
    sig_csv = re.sub(r'[\s\-]+', '_', sig_csv)                     # optional: collapse spaces/dashes to underscores

    key = csv_to_types.get((etype_csv, sig_csv))
    if not key:
        # Uncomment for debugging unknown combos:
        #print("Unmapped combo:", entity, etype_csv, sig_csv)
        continue

    current = y_true.at[entity, key]
    direction = str(row['evidenceDirection']).strip().upper()      # SUPPORTS / DOES_NOT_SUPPORT

    if direction == "SUPPORTS":
        if current == 2:              # previously DOES_NOT_SUPPORT
            y_true.at[entity, key] = 3
        elif current in (0, 1):       # NO EVIDENCE or SUPPORT only
            y_true.at[entity, key] = 1
        # leave 3 as-is
    elif direction == "DOES_NOT_SUPPORT":
        if current == 1:              # previously SUPPORT
            y_true.at[entity, key] = 3
        elif current in (0, 2):       # NO EVIDENCE or DNS only
            y_true.at[entity, key] = 2
        # leave 3 as-is

#print(len(y_true))
#print(y_true)
#sys.exit()
# Fill y_pred by reading JSON files

# target_keys = ["oncogenic_oncogenicity"]

# for ent in df['sanitized_entities'].unique():
#     mismatched = False
#     ytrue_labels = []
#     ypred_labels = []

#     for key in types_to_csv.keys():
#         if y_true.at[ent, key] != y_pred.at[ent, key]:
#             mismatched = True
#             print(ent, key)
#             print(y_true.at[ent, key], y_pred.at[ent, key])

    # if mismatched:
    #     fname = ent.replace("::", "-") + ".txt"
    #     print(f"{fname}")
    #     print(f"  y_true: {sorted(ytrue_labels)}")
    #     print(f"  y_pred: {sorted(ypred_labels)}\n")

#sys.exit()

print(y_pred)
print(y_true)

# -------------------- METRICS: PER-LABEL --------------------
per_label_rows = []
for key in types_to_csv.keys():
    yt = y_true[key].values
    yp = y_pred[key].values

    # Weighted averages across classes 0/1/2/3
    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, average="weighted", labels=[0,1,2,3], zero_division=0
    )

    per_label_rows.append({
        "label": key,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support_pos": int((yt != 0).sum()), 
        "support_total": len(yt)  # total number of samples for this key
    })

per_label_df = pd.DataFrame(per_label_rows).sort_values(
    by="support_pos", ascending=False
)
print("\n=== Per-label metrics ===")
print(per_label_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# -------------------- GLOBAL METRICS --------------------
# 1) Element-wise (micro) accuracy across all label decisions
# yt_flat = y_true.to_numpy().ravel()
# yp_flat = y_pred.to_numpy().ravel()
# overall_accuracy = accuracy_score(yt_flat, yp_flat)

# # 2) Weighted F1 across labels (weights = support of each label in y_true)
# # sklearn handles multilabel indicator matrices directly
# _, _, f1_weighted, _ = precision_recall_fscore_support(
#     y_true, y_pred, average="weighted", zero_division=0
# )

# print("\n=== Overall metrics ===")
# print(f"Accuracy (micro over all label decisions): {overall_accuracy:.3f}")
# print(f"Weighted F1 (by label support):            {f1_weighted:.3f}")

# Group keys by evidence type (prefix before first "_")
etype_groups = defaultdict(list)
for key in types_to_csv.keys():
    etype = key.split("_", 1)[0]
    etype_groups[etype].append(key)

etype_rows = []
for etype, keys in etype_groups.items():
    # Concatenate across all significance sublabels for this etype
    yt = np.concatenate([y_true[k].values for k in keys])
    yp = np.concatenate([y_pred[k].values for k in keys])

    # Weighted scores across the 4 classes
    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, average="weighted", labels=[0,1,2,3], zero_division=0
    )

    etype_rows.append({
        "etype": etype,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support_pos": int((yt != 0).sum()),   # number of non-zero (has evidence) entries
    })




etype_df = pd.DataFrame(etype_rows).sort_values(by="support_pos", ascending=False)
print("\n=== Evidence-type level metrics ===")
print(etype_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# -------------------- GLOBAL METRICS --------------------
yt_flat = y_true.to_numpy().ravel()
yp_flat = y_pred.to_numpy().ravel()

# Micro-style overall accuracy
overall_accuracy = accuracy_score(yt_flat, yp_flat)

# Weighted F1 across classes 0/1/2/3
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    yt_flat, yp_flat, average="weighted", labels=[0,1,2,3], zero_division=0
)

print("\n=== Overall metrics across all labels ===")
print(f"Accuracy (micro):   {overall_accuracy:.3f}")
print(f"Weighted Precision: {prec_w:.3f}")
print(f"Weighted Recall:    {rec_w:.3f}")
print(f"Weighted F1:        {f1_w:.3f}")



print('Positive only scores')
# ---------- Helpers for positive-only evaluation ----------
POS_CLASSES = [1, 2, 3]

def _pos_only(yt, yp):
    """
    Return yt, yp filtered to the union of positives:
    keep indices where (yt != 0) OR (yp != 0), and drop NaN preds.
    """
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)

    mask = (yt != 0) | ((~np.isnan(yp)) & (yp != 0))
    mask &= ~np.isnan(yp)          # drop missing predictions if any
    return yt[mask], yp[mask]

# -------------------- METRICS: PER-LABEL (positive-only) --------------------
per_label_rows = []
for key in types_to_csv.keys():
    yt_raw = y_true[key].values
    yp_raw = y_pred[key].values

    yt, yp = _pos_only(yt_raw, yp_raw)
    support_pos_true = int((np.asarray(yt_raw) != 0).sum())  # true positives present in dataset (for context)
    n_eval = len(yt)                                         # actually evaluated after masking

    if n_eval == 0:
        prec = rec = f1 = float("nan")
    else:
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average="weighted", labels=POS_CLASSES, zero_division=0
        )

    per_label_rows.append({
        "label": key,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support_pos_true": support_pos_true,  # count of true non-zero entries in the dataset
        "evaluated": n_eval                    # count actually evaluated after mask
    })

per_label_df = pd.DataFrame(per_label_rows).sort_values(
    by="support_pos_true", ascending=False
)
#per_label_df.to_csv('data/QA_eval_no_MCP_significance_results.csv')
print("\n=== Per-label metrics (positive-only) ===")
print(per_label_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# -------------------- EVIDENCE-TYPE METRICS (positive-only) --------------------
etype_groups = defaultdict(list)
for key in types_to_csv.keys():
    etype = key.split("_", 1)[0]
    etype_groups[etype].append(key)

etype_rows = []
for etype, keys in etype_groups.items():
    yt_concat = np.concatenate([y_true[k].values for k in keys])
    yp_concat = np.concatenate([y_pred[k].values for k in keys])

    yt, yp = _pos_only(yt_concat, yp_concat)
    support_pos_true = int((np.asarray(yt_concat) != 0).sum())
    n_eval = len(yt)

    if n_eval == 0:
        prec = rec = f1 = float("nan")
    else:
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average="weighted", labels=POS_CLASSES, zero_division=0
        )

    etype_rows.append({
        "etype": etype,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support_pos_true": support_pos_true,
        "evaluated": n_eval
    })

etype_df = pd.DataFrame(etype_rows).sort_values(by="support_pos_true", ascending=False)
print("\n=== Evidence-type level metrics (positive-only) ===")
print(etype_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# -------------------- OVERALL METRICS (positive-only) --------------------
yt_flat = y_true.to_numpy().ravel()
yp_flat = y_pred.to_numpy().ravel()

yt_pos, yp_pos = _pos_only(yt_flat, yp_flat)
n_eval_overall = len(yt_pos)

if n_eval_overall == 0:
    overall_accuracy = prec_w = rec_w = f1_w = float("nan")
else:
    overall_accuracy = accuracy_score(yt_pos, yp_pos)  # micro accuracy on positive-only window
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        yt_pos, yp_pos, average="weighted", labels=POS_CLASSES, zero_division=0
    )

print("\n=== Overall metrics across all labels (positive-only) ===")
print(f"Evaluated pairs:    {n_eval_overall}")
print(f"Accuracy (micro):   {overall_accuracy:.3f}")
print(f"Weighted Precision: {prec_w:.3f}")
print(f"Weighted Recall:    {rec_w:.3f}")
print(f"Weighted F1:        {f1_w:.3f}")