import re
import hashlib
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix


# -------------------------
# CONFIG
# -------------------------
CIVIC_CSV = Path("data") / "CIViC_clinvar_evidence_extract_2_27_26.csv"
AGENT_MODE_CSV = Path("data") / "agent_mode_run_results.csv"
QA_TRIPLET_CSV = Path("data") / "QA_triplet_dataset.csv"
CONFUSION_MATRIX_FIGURE = Path("data") / "confusion_matrices.png"

EVAL_CONFIGS = [
    {
        "name": "no_mcp",
        "display_name": "GPT",
        "kind": "txt_dir",
        "output_dir": Path("data") / "one_shot_no_mcp",
    },
    {
        "name": "mcp",
        "display_name": "GPT + MCP",
        "kind": "txt_dir",
        "output_dir": Path("data") / "one_shot_mcp",
    },
    {
        "name": "agent_mode",
        "display_name": "GPT Agent Mode",
        "kind": "csv_file",
        "csv_path": AGENT_MODE_CSV,
    },
]


# -------------------------
# NORMALIZATION HELPERS
# -------------------------
def norm_alnum_upper(x: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(x).strip().upper())


def sanitize_filename(
    name: str,
    replacement: str = "_",
    max_length: int = 255,
    ensure_unique: bool = False
) -> str:
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


# -------------------------
# THERAPY MATCHING (set-based; SUBSTITUTES = overlap with single-drug queries only)
# -------------------------
_THER_SPLIT_RE = re.compile(r"\s*(?:,|;|\||\+|/|\bAND\b|\bOR\b)\s*", flags=re.I)

@lru_cache(maxsize=200000)
def therapy_set(x: str) -> frozenset[str]:
    """
    Parse a therapy string into a normalized set of therapy tokens.
    """
    s = str(x).strip()
    if not s or s.lower() in ("none", "nan"):
        return frozenset()

    # strip common list-like wrappers
    s = s.strip().strip("[](){}")
    s = s.replace("'", "").replace('"', "")

    parts = [p.strip() for p in _THER_SPLIT_RE.split(s) if p.strip()]
    toks = []
    for p in parts:
        n = norm_alnum_upper(p)
        if n and n not in ("NONE", "NA", "NAN"):
            toks.append(n)
    return frozenset(toks)


def therapy_match_mask(rows: pd.DataFrame, qset: frozenset[str]) -> pd.Series:
    """
    Row-level therapy matching given precomputed:
      - rows["_therapy_set"] : frozenset tokens
      - rows["_tit_norm"]    : normalized interaction type

    Rules:
      - Non-SUBSTITUTES: exact set equality
      - SUBSTITUTES:
          * if query has 1 therapy -> any overlap
          * if query has >=2 therapies -> exact set equality (combo must match)
    """
    tit = rows["_tit_norm"]
    ther = rows["_therapy_set"]

    exact_mask = (tit != "SUBSTITUTES") & (ther == qset)

    subs_mask = pd.Series(False, index=rows.index)
    subs_rows = rows[tit == "SUBSTITUTES"]
    if len(subs_rows) > 0:
        if len(qset) <= 1:
            subs_mask.loc[subs_rows.index] = subs_rows["_therapy_set"].apply(lambda s: len(s & qset) > 0)
        else:
            subs_mask.loc[subs_rows.index] = (subs_rows["_therapy_set"] == qset)

    return exact_mask | subs_mask


# -------------------------
# MP SATISFACTION (AND/OR) + CODON UMBRELLA MATCHING
# -------------------------
_AND_RE = re.compile(r"\bAND\b", flags=re.I)
_OR_RE  = re.compile(r"\bOR\b",  flags=re.I)
_DIGIT_SUFFIX_RE = re.compile(r"\d+$")

def norm_mp_term(x: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(x).strip().upper())

def parse_mp(name: str):
    s = re.sub(r"\s+", " ", str(name).strip())
    if not s:
        return "SINGLE", []
    if _AND_RE.search(s):
        terms = re.split(r"\s+\bAND\b\s+", s, flags=re.I)
        return "AND", [t.strip() for t in terms if t.strip()]
    if _OR_RE.search(s):
        terms = re.split(r"\s+\bOR\b\s+", s, flags=re.I)
        return "OR", [t.strip() for t in terms if t.strip()]
    return "SINGLE", [s]

def term_match(q_term: str, e_term: str, allow_positional_query_umbrella: bool = True) -> bool:
    """
    True if query term matches evidence term, allowing:
      - exact normalized match
      - OPTIONAL directional codon umbrella match:
          query positional (EZH2Y646) can match evidence specific (EZH2Y646S),
          but query specific (EZH2Y646S) does NOT match evidence positional (EZH2Y646).
    """
    qn = norm_mp_term(q_term)
    en = norm_mp_term(e_term)
    if not qn or not en:
        return False

    if qn == en:
        return True

    if allow_positional_query_umbrella:
        if _DIGIT_SUFFIX_RE.search(qn) and en.startswith(qn):
            suffix = en[len(qn):]
            if re.fullmatch(r"[A-Z]+", suffix):
                return True

    return False

@lru_cache(maxsize=200000)
def mp_satisfied(query_mp_name: str, evidence_mp_name: str) -> bool:
    """
    Query-driven MP satisfaction:

    - If QUERY is AND:
        require EVIDENCE is AND and the AND profiles match as a whole (mutual coverage).
    - If QUERY is OR:
        allow evidence SINGLE/OR if any overlap; do NOT allow evidence AND.
    - If QUERY is SINGLE:
        allow evidence SINGLE/OR if any term matches; do NOT allow evidence AND.
    """
    q_op, q_terms = parse_mp(query_mp_name)
    e_op, e_terms = parse_mp(evidence_mp_name)

    if not q_terms or not e_terms:
        return False

    if q_op == "AND":
        if e_op != "AND":
            return False
        q_covered = all(any(term_match(qt, et) for et in e_terms) for qt in q_terms)
        e_covered = all(any(term_match(qt, et) for qt in q_terms) for et in e_terms)
        return q_covered and e_covered

    if q_op == "OR":
        if e_op == "AND":
            return False
        return any(term_match(qt, et) for qt in q_terms for et in e_terms)

    if e_op == "AND":
        return False
    q0 = q_terms[0]
    return any(term_match(q0, et) for et in e_terms)


# -------------------------
# LABEL SPACE (Task 2 only; 23 lines)
# -------------------------
ITEMS = [
    ("predictive",   "sensitivity_response"),
    ("predictive",   "resistance"),
    ("predictive",   "adverse_response"),
    ("predictive",   "reduced_sensitivity"),
    ("predictive",   "N/A"),

    ("prognostic",   "better_outcome"),
    ("prognostic",   "poor_outcome"),
    ("prognostic",   "N/A"),

    ("diagnostic",   "positive"),
    ("diagnostic",   "negative"),

    ("predisposing", "predisposition"),
    ("predisposing", "protectiveness"),
    ("predisposing", "uncertain_significance"),
    ("predisposing", "N/A"),

    ("oncogenic",    "oncogenicity"),
    ("oncogenic",    "protectiveness"),
    ("oncogenic",    "N/A"),

    ("functional",   "gain_of_function"),
    ("functional",   "loss_of_function"),
    ("functional",   "unaltered_function"),
    ("functional",   "neomorphic"),
    ("functional",   "dominant_negative"),
    ("functional",   "unknown"),
]
ALL_LABEL_KEYS = [f"{t}_{s}" for t, s in ITEMS]

types_to_csv = {
    # Predictive
    "predictive_sensitivity_response": ("PREDICTIVE", "SENSITIVITYRESPONSE"),
    "predictive_resistance":           ("PREDICTIVE", "RESISTANCE"),
    "predictive_adverse_response":     ("PREDICTIVE", "ADVERSE_RESPONSE"),
    "predictive_reduced_sensitivity":  ("PREDICTIVE", "REDUCED_SENSITIVITY"),
    "predictive_N/A":                  ("PREDICTIVE", "N/A"),

    # Prognostic
    "prognostic_better_outcome":       ("PROGNOSTIC", "BETTER_OUTCOME"),
    "prognostic_poor_outcome":         ("PROGNOSTIC", "POOR_OUTCOME"),
    "prognostic_N/A":                  ("PROGNOSTIC", "N/A"),

    # Diagnostic
    "diagnostic_positive":             ("DIAGNOSTIC", "POSITIVE"),
    "diagnostic_negative":             ("DIAGNOSTIC", "NEGATIVE"),

    # Predisposing
    "predisposing_predisposition":         ("PREDISPOSING", "PREDISPOSITION"),
    "predisposing_protectiveness":         ("PREDISPOSING", "PROTECTIVENESS"),
    "predisposing_uncertain_significance": ("PREDISPOSING", "UNCERTAIN_SIGNIFICANCE"),
    "predisposing_N/A":                    ("PREDISPOSING", "N/A"),

    # Oncogenic
    "oncogenic_oncogenicity":          ("ONCOGENIC", "ONCOGENICITY"),
    "oncogenic_protectiveness":        ("ONCOGENIC", "PROTECTIVENESS"),
    "oncogenic_N/A":                   ("ONCOGENIC", "N/A"),

    # Functional
    "functional_gain_of_function":     ("FUNCTIONAL", "GAIN_OF_FUNCTION"),
    "functional_loss_of_function":     ("FUNCTIONAL", "LOSS_OF_FUNCTION"),
    "functional_unaltered_function":   ("FUNCTIONAL", "UNALTERED_FUNCTION"),
    "functional_neomorphic":           ("FUNCTIONAL", "NEOMORPHIC"),
    "functional_dominant_negative":    ("FUNCTIONAL", "DOMINANT_NEGATIVE"),
    "functional_unknown":              ("FUNCTIONAL", "UNKNOWN"),
}

csv_to_types_norm = {
    (norm_alnum_upper(et), norm_alnum_upper(sig)): key
    for key, (et, sig) in types_to_csv.items()
}


def _clean_none(x: str) -> str:
    x = str(x).strip()
    return "None" if x == "" or x.lower() in ("none", "nan") else x


def build_filter_rows_for_entity(df_all: pd.DataFrame, entity_meta: pd.DataFrame):
    def filter_rows_for_entity(ent: str) -> pd.DataFrame:
        """
        Context filters + MP satisfaction (truth construction):

        - If disease is specified (not 'None'): require disease match.
        - If disease is 'None': require disease_name == 'None'.
        - If therapy is specified: require therapy match (set-based; SUBSTITUTES special-case).
        - If therapy is 'None': require therapies == 'None'.
        - Require molecular profile satisfaction using AND/OR + codon umbrella matching.
        """
        if ent not in entity_meta.index:
            return df_all.iloc[0:0]

        q_mp = entity_meta.at[ent, "molecularProfile_name"]
        q_dz = _clean_none(entity_meta.at[ent, "disease_name"])
        q_tx = _clean_none(entity_meta.at[ent, "therapies"])

        rows = df_all

        # Disease filter
        if isinstance(q_dz, str) and q_dz != "None":
            rows = rows[rows["disease_name"] == q_dz]
        else:
            rows = rows[rows["disease_name"] == "None"]

        # Therapy filter
        if isinstance(q_tx, str) and q_tx != "None":
            qset = therapy_set(q_tx)
            rows = rows[therapy_match_mask(rows, qset)]
        else:
            rows = rows[rows["therapies"] == "None"]

        # Molecular profile satisfaction
        rows = rows[rows["molecularProfile_name"].apply(lambda mp: mp_satisfied(q_mp, mp))]

        return rows

    return filter_rows_for_entity


def build_ground_truth_matrix(entities, filter_rows_for_entity) -> pd.DataFrame:
    """
    Builds the Task 2 ground-truth matrix (bitmask encoding) for a list of entities.

    For each entity / significance-key cell:
      0       = no matching CIViC evidence rows (-> "C" / No Evidence)
      bit 1   = some matching evidence row SUPPORTS   (-> contributes "A")
      bit 2   = some matching evidence row DOES NOT SUPPORT (-> contributes "B")
      3 (1|2) = both bits set, OR a row with unknown/NA direction (-> ambiguous "A,B")
    """
    y_true = pd.DataFrame(0, index=entities, columns=ALL_LABEL_KEYS, dtype=int)

    for ent in entities:
        rows = filter_rows_for_entity(ent)

        for _, row in rows.iterrows():
            et = norm_alnum_upper(row.get("evidenceType", ""))
            sig = norm_alnum_upper(row.get("significance", ""))
            key = csv_to_types_norm.get((et, sig))
            if not key:
                continue

            cur = int(y_true.at[ent, key])

            d_raw = row.get("evidenceDirection", "")
            d = norm_alnum_upper("" if pd.isna(d_raw) else d_raw)

            # Direction unknown/NA -> ambiguous => A,B
            if d in ("NA", "NAN", ""):
                cur |= 3
            elif d == "SUPPORTS":
                cur |= 1
            elif d == "DOESNOTSUPPORT":
                cur |= 2

            y_true.at[ent, key] = cur

    return y_true


_TIME_RE = re.compile(
    r"###\s*TIME\s*\(Seconds\)\s*###\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.IGNORECASE
)

def _norm_sig_label_from_model(sig_raw: str) -> str:
    s = str(sig_raw).strip()
    if s.upper().replace(" ", "") in ("N/A", "NA"):
        return "N/A"
    s = s.lower().strip()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s

def parse_task2_outputs(text: str):
    """
    Returns: (task2_dict, time_seconds_or_None)

    Robust to extra logging; if '### LLM OUTPUT ###' exists, parses after it.
    Ignores anything after '### TIME' marker for output parsing.
    """
    m_time = _TIME_RE.search(text)
    time_s = float(m_time.group(1)) if m_time else None

    if "### LLM OUTPUT ###" in text:
        text = text.split("### LLM OUTPUT ###", 1)[1]

    if "### TIME" in text:
        text = text.split("### TIME", 1)[0]

    task2 = {k: None for k in ALL_LABEL_KEYS}

    r2 = re.compile(
        r"^(predictive|diagnostic|prognostic|predisposing|oncogenic|functional)\s*[—–-]\s*([^:]+?)\s*:\s*([ABC](?:\s*,\s*[ABC])?)\s*$",
        re.MULTILINE
    )
    for m in r2.finditer(text):
        etype = m.group(1).strip().lower()
        sig = _norm_sig_label_from_model(m.group(2))
        val = m.group(3).replace(" ", "").upper()
        if val in ("B,A", "A,B"):
            val = "A,B"

        key = f"{etype}_{sig}"
        if key in task2:
            task2[key] = val

    return task2, time_s


def task2_label_to_int(val: str) -> int:
    # 0=C (No Evidence), 1=A, 2=B, 3=A,B, 4=INVALID/PARSE ERROR
    if val is None:
        return 4
    v = str(val).strip().upper().replace(" ", "")
    if v == "C":
        return 0
    if v == "A":
        return 1
    if v == "B":
        return 2
    if v in ("A,B", "B,A"):
        return 3
    return 4


def load_eval_records(cfg: dict) -> pd.DataFrame:
    """
    Returns a standardized DataFrame with: source_id, sanitized_entities, raw_output.
    """
    kind = cfg["kind"]

    if kind == "txt_dir":
        output_dir = cfg["output_dir"]
        output_files = sorted(output_dir.glob("*.txt"))
        rows = []

        for p in output_files:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            rows.append({
                "source_id": p.name,
                "sanitized_entities": sanitize_filename(p.stem, replacement="_"),
                "raw_output": raw,
            })

        return pd.DataFrame(rows)

    if kind == "csv_file":
        csv_path = cfg["csv_path"]
        df = pd.read_csv(csv_path)

        required = {"MP", "disease", "therapy", "output"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        df = df.copy()
        df["MP"] = df["MP"].astype(str)
        df["disease"] = df["disease"].fillna("None").astype(str)
        df["therapy"] = df["therapy"].fillna("None").astype(str)
        df["raw_output"] = df["output"].fillna("").astype(str)

        df["entities"] = df["MP"] + "_" + df["disease"] + "_" + df["therapy"]
        df["sanitized_entities"] = df["entities"].apply(lambda x: sanitize_filename(x, replacement="_"))

        return (
            df[["sanitized_entities", "raw_output"]]
            .reset_index()
            .rename(columns={"index": "source_id"})
        )

    raise ValueError(f"Unknown eval kind: {kind}")


# -------------------------
# CONFUSION MATRIX FIGURE
# -------------------------
# Row/column order, top-to-bottom and left-to-right: A, B, A&B, C.
CM_LABEL_CODES = [1, 2, 3, 0]
CM_LABEL_TEXT = [
    "Supports",
    "Does not\nSupport",
    "Contradicting",
    "No\nEvidence",
]


def score_config_confusion_matrix(cfg: dict, df_all: pd.DataFrame) -> np.ndarray:
    """
    Loads a config's raw model outputs, builds the matching ground-truth + prediction
    matrices (same entity-matching logic as the eval pipeline), and returns the
    5x5 confusion matrix (rows=true, cols=predicted) over EVERY entity/significance
    pair -- no positive-only filtering, so the "no evidence" class is included.
    """
    records = load_eval_records(cfg)
    if records.empty:
        return np.zeros((len(CM_LABEL_CODES), len(CM_LABEL_CODES)), dtype=int)

    output_entities = set(records["sanitized_entities"])
    df_q = df_all[df_all["sanitized_entities"].isin(output_entities)].copy()
    scored_entities = sorted(set(df_q["sanitized_entities"]).intersection(output_entities))
    if not scored_entities:
        return np.zeros((len(CM_LABEL_CODES), len(CM_LABEL_CODES)), dtype=int)

    entity_meta = (
        df_q.groupby("sanitized_entities")[["molecularProfile_name", "disease_name", "therapies"]]
            .first()
    )
    filter_rows_for_entity = build_filter_rows_for_entity(df_all, entity_meta)
    y_true = build_ground_truth_matrix(scored_entities, filter_rows_for_entity)

    y_pred = pd.DataFrame(4, index=scored_entities, columns=ALL_LABEL_KEYS, dtype=int)
    records_by_ent = (
        records.drop_duplicates(subset=["sanitized_entities"], keep="first")
               .set_index("sanitized_entities")
    )

    for ent in scored_entities:
        if ent not in records_by_ent.index:
            continue
        t2, _ = parse_task2_outputs(records_by_ent.at[ent, "raw_output"])
        for k in ALL_LABEL_KEYS:
            y_pred.at[ent, k] = task2_label_to_int(t2.get(k))

    yt_flat = y_true.to_numpy().ravel()
    yp_flat = y_pred.to_numpy().ravel()

    return confusion_matrix(yt_flat, yp_flat, labels=CM_LABEL_CODES)


def build_confusion_matrix_figure(eval_configs: list, df_all: pd.DataFrame, out_path: Path) -> None:
    """
    Builds one figure with a confusion matrix per eval config (left-to-right in the
    order given), all sharing one red<->green color scale (red = incorrect, green =
    correct; more samples = more saturated), with a single horizontal colorbar
    spanning the bottom of the figure.
    """
    n = len(CM_LABEL_CODES)

    cms = [score_config_confusion_matrix(cfg, df_all) for cfg in eval_configs]
    titles = [cfg["display_name"] for cfg in eval_configs]

    # Signed matrices: +count on the diagonal (correct), -count off-diagonal (incorrect)
    signed_list = []
    for cm in cms:
        signed = cm.astype(float).copy()
        off_diag = ~np.eye(n, dtype=bool)
        signed[off_diag] *= -1
        signed_list.append(signed)

    max_count = max(1, max(int(cm.max()) for cm in cms))

    # Symmetric log scale so the (typically huge) "No Evidence" diagonal cell doesn't
    # wash out every other cell, while 0 samples stays exactly white at the center.
    norm = mcolors.SymLogNorm(linthresh=1, vmin=-max_count, vmax=max_count)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_white_green", ["#8B0000", "#FFFFFF", "#1B5E20"]
    )

    fig = plt.figure(figsize=(6 * len(eval_configs), 7))
    gs = fig.add_gridspec(1, len(eval_configs), hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(eval_configs))]

    im = None
    for idx, (ax, cm, signed, title) in enumerate(zip(axes, cms, signed_list, titles)):
        im = ax.imshow(signed, cmap=cmap, norm=norm, aspect="equal")
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(CM_LABEL_TEXT, rotation=0, ha="center", fontsize=9)
        ax.set_yticklabels(CM_LABEL_TEXT, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        if idx == 0:
            ax.set_ylabel("True (CIViC ground truth)", fontsize=10)

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#999999", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        for i in range(n):
            for j in range(n):
                count = int(cm[i, j])
                norm_val = norm(signed[i, j])
                text_color = "white" if abs(norm_val - 0.5) > 0.32 else "black"
                ax.text(j, i, str(count), ha="center", va="center", color=text_color, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix figure: {out_path}")


def load_used_entities(csv_path: Path) -> pd.DataFrame:
    """
    Loads the agent-mode prompts CSV and reconstructs sanitized_entities exactly the
    way eval_one_shot.py's load_eval_records (kind="csv_file") does, so the resulting
    entity IDs line up 1:1 with df_all["sanitized_entities"].

    Returns a DataFrame with columns: MP, disease, therapy, sanitized_entities
    (one row per row of the input CSV; duplicates are NOT collapsed here).
    """
    df = pd.read_csv(csv_path)

    required = {"MP", "disease", "therapy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["MP"] = df["MP"].astype(str)
    df["disease"] = df["disease"].fillna("None").astype(str)
    df["therapy"] = df["therapy"].fillna("None").astype(str)

    df["entities"] = df["MP"] + "_" + df["disease"] + "_" + df["therapy"]
    df["sanitized_entities"] = df["entities"].apply(lambda x: sanitize_filename(x, replacement="_"))

    return df[["MP", "disease", "therapy", "sanitized_entities"]]


def build_qa_triplet_dataset(df_all: pd.DataFrame, out_path: Path, restrict_to_entities=None) -> pd.DataFrame:
    """
    Builds the ground-truth QA dataset: one row per unique (MP, Disease, Therapy)
    triplet found in the CIViC extract, along with the comma-separated list of
    evidenceType_significance labels for which CIViC evidence:
      - Evidence_Supports      (A)      -> evidence supports, no opposing evidence
      - Evidence_Opposes       (B)      -> evidence opposes, no supporting evidence
      - Contradicting_Evidence (A & B)  -> evidence both supports AND opposes (or direction unknown/N/A)
      - No_Evidence            (C)      -> no matching CIViC evidence at all

    Each label falls into exactly one of these four mutually-exclusive categories.

    If restrict_to_entities is given (an iterable of sanitized_entities values), the
    output is limited to just those triplets instead of every triplet in df_all.
    """
    entity_meta = (
        df_all.groupby("sanitized_entities")[["molecularProfile_name", "disease_name", "therapies"]]
              .first()
    )

    if restrict_to_entities is not None:
        entity_meta = entity_meta.loc[entity_meta.index.intersection(restrict_to_entities)]

    all_entities = list(entity_meta.index)

    filter_rows_for_entity = build_filter_rows_for_entity(df_all, entity_meta)
    y_true = build_ground_truth_matrix(all_entities, filter_rows_for_entity)

    def _to_display(x: str) -> str:
        x = str(x).strip()
        return "" if x == "" or x.lower() in ("none", "nan") else x

    rows = []
    for ent in all_entities:
        cur_row = y_true.loc[ent]

        evidence_supports = [k for k in ALL_LABEL_KEYS if int(cur_row[k]) == 1]
        evidence_opposes = [k for k in ALL_LABEL_KEYS if int(cur_row[k]) == 2]
        contradicting_evidence = [k for k in ALL_LABEL_KEYS if int(cur_row[k]) == 3]
        no_evidence = [k for k in ALL_LABEL_KEYS if int(cur_row[k]) == 0]

        rows.append({
            "MP": entity_meta.at[ent, "molecularProfile_name"],
            "Disease": _to_display(entity_meta.at[ent, "disease_name"]),
            "Therapy": _to_display(entity_meta.at[ent, "therapies"]),
            "Evidence_Supports": ",".join(evidence_supports),
            "Evidence_Opposes": ",".join(evidence_opposes),
            "Contradicting_Evidence": ",".join(contradicting_evidence),
            "No_Evidence": ",".join(no_evidence),
            "entity_id": ent,
        })

    qa_df = pd.DataFrame(rows, columns=[
        "MP", "Disease", "Therapy",
        "Evidence_Supports", "Evidence_Opposes", "Contradicting_Evidence", "No_Evidence",
        "entity_id",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    qa_df.to_csv(out_path, index=False)

    print(f"Saved QA triplet dataset: {out_path}")
    print(f"  Unique MP/Disease/Therapy triplets: {len(qa_df)}")

    return qa_df


# -------------------------
# LOAD DATA + BUILD/SAVE QA TRIPLET DATASET
# -------------------------
if __name__ == "__main__":
    df_all = pd.read_csv(CIVIC_CSV)

    # Normalize missing/empty significance to N/A
    df_all["significance"] = df_all["significance"].replace("", np.nan).fillna("N/A")

    df_all["therapies"] = df_all["therapies"].replace("", np.nan).fillna("None")
    df_all["disease_name"] = df_all["disease_name"].replace("", np.nan).fillna("None")

    # normalize therapy interaction type
    df_all["_tit_norm"] = df_all["therapy_interaction_type"].fillna("").astype(str).apply(norm_alnum_upper)

    # precompute therapy sets for matching
    df_all["_therapy_set"] = df_all["therapies"].astype(str).apply(therapy_set)

    df_all["entities"] = (
        df_all["molecularProfile_name"].astype(str) + "_" +
        df_all["disease_name"].astype(str) + "_" +
        df_all["therapies"].astype(str)
    )
    df_all["sanitized_entities"] = df_all["entities"].apply(lambda x: sanitize_filename(x, replacement="_"))

    # -------------------------
    # RESTRICT TO ONLY THE TRIPLETS ACTUALLY USED IN THE EVAL
    # -------------------------
    used = load_used_entities(AGENT_MODE_CSV)
    used_entities = set(used["sanitized_entities"])

    csv_entities = set(df_all["sanitized_entities"])
    matched_entities = used_entities & csv_entities
    unmatched_entities = used_entities - csv_entities

    print(f"Rows in agent-mode prompts CSV: {len(used)}")
    print(f"Unique triplets referenced:     {len(used_entities)}")
    print(f"Matched against CIViC extract:  {len(matched_entities)}")

    if unmatched_entities:
        print("\nWARNING: the following triplets from the agent-mode CSV have no")
        print("matching entity in the CIViC extract and will be EXCLUDED:")
        missing_rows = used[used["sanitized_entities"].isin(unmatched_entities)]
        for _, r in missing_rows.drop_duplicates(subset=["sanitized_entities"]).iterrows():
            print(f"    MP={r['MP']!r}  disease={r['disease']!r}  therapy={r['therapy']!r}")

    build_qa_triplet_dataset(df_all, QA_TRIPLET_CSV, restrict_to_entities=matched_entities)

    # -------------------------
    # BUILD + SAVE CONFUSION MATRIX FIGURE (GPT / GPT + MCP / GPT Agent Mode)
    # -------------------------
    build_confusion_matrix_figure(EVAL_CONFIGS, df_all, CONFUSION_MATRIX_FIGURE)