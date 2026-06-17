import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# -------------------------
# CONFIG
# -------------------------
CIVIC_CSV = Path("data") / "CIViC_clinvar_evidence_extract_2_27_26.csv"
AGENT_MODE_CSV = Path("data") / "agent_mode_run_results.csv"

EVAL_CONFIGS = [
    {
        "name": "no_mcp",
        "kind": "txt_dir",
        "output_dir": Path("data") / "one_shot_no_mcp",
        "out_metrics": Path("data") / "one_shot_no_mcp_eval",
    },
    {
        "name": "mcp",
        "kind": "txt_dir",
        "output_dir": Path("data") / "one_shot_mcp",
        "out_metrics": Path("data") / "one_shot_mcp_eval",
    },
    {
        "name": "agent_mode",
        "kind": "csv_file",
        "csv_path": AGENT_MODE_CSV,
        "out_metrics": Path("data") / "agent_mode_eval",
    },
]

debug_misses = False
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


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


def pos_only_window(yt, yp, pos_classes):
    """
    Positive-only window:
      keep indices where true is positive (yt != 0) OR predicted is positive (yp in pos_classes)
    """
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    pred_pos = np.isin(yp, pos_classes)
    mask = (yt != 0) | pred_pos
    return yt[mask], yp[mask]


def mean_ci95_bootstrap(values, n_resamples=10000, seed=42):
    """
    Percentile bootstrap 95% CI for the mean.
    Returns: (mean, ci_low, ci_high)
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean_val = float(arr.mean())

    if arr.size == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[sample_idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return mean_val, float(ci_low), float(ci_high)


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
# LABEL SPACE / PROMPT ORDER (Task 2 only; 23 lines)
# -------------------------
POS_CLASSES = [1, 2, 3]  # A / B / A,B

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


# -------------------------
# PARSING MODEL OUTPUT (+ TIME)
# -------------------------
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
    # 0=C (No Evidence), 1=A, 2=B, 3=A,B, 4=INVALID/UNKNOWN
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


def load_eval_records(cfg: dict) -> pd.DataFrame:
    """
    Returns a standardized DataFrame with:
      - source_id
      - sanitized_entities
      - raw_output
      - time_seconds
    """
    kind = cfg["kind"]

    if kind == "txt_dir":
        output_dir = cfg["output_dir"]
        output_files = sorted(output_dir.glob("*.txt"))
        rows = []

        for p in output_files:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            _, parsed_time = parse_task2_outputs(raw)
            rows.append({
                "source_id": p.name,
                "sanitized_entities": sanitize_filename(p.stem, replacement="_"),
                "raw_output": raw,
                "time_seconds": parsed_time,
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

        # Prefer explicit runtime column if present
        if "seconds" in df.columns:
            df["time_seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
        else:
            df["time_seconds"] = np.nan

        # Fallback to parsing a TIME block if needed
        parsed_times = df["raw_output"].apply(lambda x: parse_task2_outputs(x)[1])
        df["time_seconds"] = df["time_seconds"].where(df["time_seconds"].notna(), parsed_times)

        df["entities"] = df["MP"] + "_" + df["disease"] + "_" + df["therapy"]
        df["sanitized_entities"] = df["entities"].apply(lambda x: sanitize_filename(x, replacement="_"))

        return (
            df[["sanitized_entities", "raw_output", "time_seconds"]]
            .reset_index()
            .rename(columns={"index": "source_id"})
        )

    raise ValueError(f"Unknown eval kind: {kind}")


def evaluate_config(cfg: dict, df_all: pd.DataFrame) -> dict:
    eval_name = cfg["name"]
    out_metrics = cfg["out_metrics"]
    out_metrics.mkdir(parents=True, exist_ok=True)

    source_desc = str(cfg.get("output_dir", cfg.get("csv_path", "")))

    print("\n" + "=" * 100)
    print(f"Evaluating: {eval_name}")
    print(f"  Source      = {source_desc}")
    print(f"  OUT_METRICS = {out_metrics}")

    records = load_eval_records(cfg)
    if records.empty:
        print("  No evaluation records found; skipping.")
        return {
            "eval_name": eval_name,
            "source": source_desc,
            "entities_scored": 0,
            "evaluated_pairs_positive_window": 0,
            "overall_accuracy_positive_window": float("nan"),
            "overall_micro_precision_pos_classes": float("nan"),
            "overall_micro_recall_pos_classes": float("nan"),
            "overall_micro_f1_pos_classes": float("nan"),
            "runs_with_time": 0,
            "avg_time_seconds": float("nan"),
            "avg_time_seconds_ci95_low": float("nan"),
            "avg_time_seconds_ci95_high": float("nan"),
        }

    # Duplicates / collisions
    c = Counter(records["sanitized_entities"])
    dups = {k: v for k, v in c.items() if v > 1}

    print("  Records:", len(records))
    print("  Unique sanitized entities:", records["sanitized_entities"].nunique())

    if dups:
        print("\n  SANITIZE COLLISIONS / DUPLICATE ENTITIES:")
        for k, v in sorted(dups.items(), key=lambda kv: -kv[1]):
            print(f"    {k}  <-- {v} records")

    output_entities = set(records["sanitized_entities"])

    df_q = df_all[df_all["sanitized_entities"].isin(output_entities)].copy()
    scored_entities = sorted(set(df_q["sanitized_entities"]).intersection(output_entities))
    if not scored_entities:
        print("  No overlap between evaluation records and CSV entities; skipping.")
        return {
            "eval_name": eval_name,
            "source": source_desc,
            "entities_scored": 0,
            "evaluated_pairs_positive_window": 0,
            "overall_accuracy_positive_window": float("nan"),
            "overall_micro_precision_pos_classes": float("nan"),
            "overall_micro_recall_pos_classes": float("nan"),
            "overall_micro_f1_pos_classes": float("nan"),
            "runs_with_time": 0,
            "avg_time_seconds": float("nan"),
            "avg_time_seconds_ci95_low": float("nan"),
            "avg_time_seconds_ci95_high": float("nan"),
        }

    # Check records with no CSV entity match
    csv_entities = set(df_all["sanitized_entities"])
    missing_in_csv = sorted(output_entities - csv_entities)
    if missing_in_csv:
        print("\n  RECORDS WITH NO MATCHING CSV ENTITY (won't be scored):")
        for k in missing_in_csv:
            print("   ", k)

    entity_meta = (
        df_q.groupby("sanitized_entities")[["molecularProfile_name", "disease_name", "therapies"]]
          .first()
    )
    filter_rows_for_entity = build_filter_rows_for_entity(df_all, entity_meta)

    # -------------------------
    # BUILD TASK 2 GROUND TRUTH
    # -------------------------
    y_true = pd.DataFrame(0, index=scored_entities, columns=ALL_LABEL_KEYS, dtype=int)

    for ent in scored_entities:
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

    # -------------------------
    # PARSE PREDICTIONS (+ COLLECT TIMES)
    # -------------------------
    y_pred = pd.DataFrame(4, index=scored_entities, columns=ALL_LABEL_KEYS, dtype=int)

    # If duplicates exist, keep first occurrence
    records_by_ent = (
        records.drop_duplicates(subset=["sanitized_entities"], keep="first")
               .set_index("sanitized_entities")
    )

    times_s = []

    for ent in scored_entities:
        if ent not in records_by_ent.index:
            continue

        raw = records_by_ent.at[ent, "raw_output"]
        t2, parsed_time = parse_task2_outputs(raw)

        stored_time = records_by_ent.at[ent, "time_seconds"]
        tsec = stored_time if pd.notna(stored_time) else parsed_time

        if tsec is not None and pd.notna(tsec):
            times_s.append(float(tsec))

        for k in ALL_LABEL_KEYS:
            y_pred.at[ent, k] = task2_label_to_int(t2.get(k))

    # -------------------------
    # METRICS (POSITIVE-ONLY, MICRO) — MAIN TABLE BY EVIDENCE TYPE
    # -------------------------
    etype_groups = defaultdict(list)
    for key in ALL_LABEL_KEYS:
        etype_groups[key.split("_", 1)[0]].append(key)

    etype_rows = []
    for etype, keys in etype_groups.items():
        yt_concat = np.concatenate([y_true[k].values for k in keys])
        yp_concat = np.concatenate([y_pred[k].values for k in keys])

        samples = int((np.asarray(yt_concat) != 0).sum())
        yt_win, yp_win = pos_only_window(yt_concat, yp_concat, POS_CLASSES)
        evaluated = len(yt_win)

        if evaluated == 0:
            prec = rec = f1 = float("nan")
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                yt_win, yp_win,
                average="micro",
                labels=POS_CLASSES,
                zero_division=0
            )

        etype_rows.append({
            "Evaluation": eval_name,
            "Evidence Type": etype.capitalize(),
            "Samples": samples,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Evaluated": evaluated,
        })

    etype_df = pd.DataFrame(etype_rows).sort_values(by="Samples", ascending=False)
    etype_df.to_csv(out_metrics / "evidence_type_positive_micro.csv", index=False)

    # -------------------------
    # METRICS (POSITIVE-ONLY, MICRO) — SUPPLEMENTAL TABLE BY SIGNIFICANCE ITEM
    # -------------------------
    per_label_rows = []
    for key in ALL_LABEL_KEYS:
        yt_raw = y_true[key].values
        yp_raw = y_pred[key].values

        samples = int((np.asarray(yt_raw) != 0).sum())
        yt_win, yp_win = pos_only_window(yt_raw, yp_raw, POS_CLASSES)
        evaluated = len(yt_win)

        if evaluated == 0:
            prec = rec = f1 = float("nan")
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                yt_win, yp_win,
                average="micro",
                labels=POS_CLASSES,
                zero_division=0
            )

        etype = key.split("_", 1)[0]
        sig = key.split("_", 1)[1]
        per_label_rows.append({
            "Evaluation": eval_name,
            "Label": key,
            "Evidence Type": etype.capitalize(),
            "Significance": sig,
            "Samples": samples,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Evaluated": evaluated,
        })

    per_label_df = pd.DataFrame(per_label_rows).sort_values(by="Samples", ascending=False)
    per_label_df.to_csv(out_metrics / "significance_positive_micro.csv", index=False)

    # -------------------------
    # OVERALL SUMMARY (POSITIVE-ONLY WINDOW)
    # -------------------------
    yt_flat = y_true.to_numpy().ravel()
    yp_flat = y_pred.to_numpy().ravel()

    yt_win, yp_win = pos_only_window(yt_flat, yp_flat, POS_CLASSES)
    evaluated_overall = len(yt_win)

    if evaluated_overall == 0:
        overall_acc = overall_p = overall_r = overall_f1 = float("nan")
    else:
        overall_acc = accuracy_score(yt_win, yp_win)
        overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
            yt_win, yp_win,
            average="micro",
            labels=POS_CLASSES,
            zero_division=0
        )

    avg_time, avg_time_ci_low, avg_time_ci_high = mean_ci95_bootstrap(
        times_s,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    summary_row = {
        "eval_name": eval_name,
        "source": source_desc,
        "entities_scored": len(scored_entities),
        "evaluated_pairs_positive_window": evaluated_overall,
        "overall_accuracy_positive_window": overall_acc,
        "overall_micro_precision_pos_classes": overall_p,
        "overall_micro_recall_pos_classes": overall_r,
        "overall_micro_f1_pos_classes": overall_f1,
        "runs_with_time": int(len(times_s)),
        "avg_time_seconds": avg_time,
        "avg_time_seconds_ci95_low": avg_time_ci_low,
        "avg_time_seconds_ci95_high": avg_time_ci_high,
    }
    pd.DataFrame([summary_row]).to_csv(out_metrics / "summary_positive_micro.csv", index=False)

    # -------------------------
    # PRINT
    # -------------------------
    print(f"  Entities scored: {len(scored_entities)}")

    if len(times_s) > 0:
        print(
            "  Average time (seconds): "
            f"{avg_time:.3f} "
            f"[95% CI {avg_time_ci_low:.3f}, {avg_time_ci_high:.3f}] "
            f"(n={len(times_s)})"
        )
    else:
        print("  Average time (seconds): NA  (no TIME blocks found / no seconds column)")

    print("\n=== Evidence-type table (positive-only, micro over A/B/A,B) ===")
    print(etype_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Overall (positive-only window) ===")
    print(f"Evaluated pairs: {evaluated_overall}")
    print(f"Accuracy (exact match, pos-window):  {overall_acc:.3f}")
    print(f"Micro Precision/Recall/F1 (A/B/A,B): {overall_p:.3f} / {overall_r:.3f} / {overall_f1:.3f}")

    print("\nSaved:")
    print(f"  {out_metrics / 'evidence_type_positive_micro.csv'}")
    print(f"  {out_metrics / 'significance_positive_micro.csv'}")
    print(f"  {out_metrics / 'summary_positive_micro.csv'}")

    return summary_row


# -------------------------
# LOAD DATA ONCE
# -------------------------
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
# RUN ALL EVALUATIONS
# -------------------------
all_summaries = []
for cfg in EVAL_CONFIGS:
    all_summaries.append(evaluate_config(cfg, df_all))

combined_summary = pd.DataFrame(all_summaries)
combined_summary.to_csv(Path("data") / "combined_eval_summary.csv", index=False)

print("\n" + "=" * 100)
print("Combined summary:")
print(combined_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nSaved: {Path('data') / 'combined_eval_summary.csv'}")