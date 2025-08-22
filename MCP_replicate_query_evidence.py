#!/usr/bin/env python
"""
query_evidence.py

Call the `get_variant_evidence` tool on the CIViC MCP server from the command line.

Examples
--------
python query_evidence.py --mp "EGFR" \
                         --disease "Lung Non-small Cell Carcinoma" \
                         --therapy "Erlotinib"
"""
import argparse
import asyncio
import logging
from typing import Dict
import re
import unicodedata
from collections import Counter

import pandas as pd
import time
import statistics
import numpy as np
import json

from mcp import ClientSession
from mcp.client.sse import sse_client   # use streamable_http_client if the server migrates

MCP_SSE_URL = "https://civic-mcp-server.larscivic.workers.dev/sse"

with open("./src/data/molecular_profile_map.json", "r") as f:
    MP_map = json.load(f)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
# )

# -----------------------------------------------------------------------------

def normalize_str(s: str) -> str:
    """
    Decompose accents, strip diacritics, lowercase, remove non-alphanumerics,
    collapse whitespace, and trim.
    """
    # 1. Decompose accents (NFKD) and strip combining marks
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # 2. Lowercase
    s = s.lower()
    # 3. Keep only letters, digits, spaces
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # 4. Collapse runs of whitespace
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def dice_coefficient(a: str, b: str) -> float:
    """
    Compute Sørensen–Dice coefficient between two strings based on bigrams.
    Returns 1.0 for exact match on <2‑char strings, 0.0 if either is empty.
    """
    if not a or not b:
        return 0.0
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else 0.0

    # Build bigram multisets
    ba = [a[i:i+2] for i in range(len(a) - 1)]
    bb = [b[i:i+2] for i in range(len(b) - 1)]
    ca, cb = Counter(ba), Counter(bb)

    # Intersection size
    intersection = sum(min(ca[gram], cb[gram]) for gram in ca)
    total = sum(ca.values()) + sum(cb.values())

    return (2.0 * intersection) / total if total > 0 else 0.0

def normalize_entity(name, lookup, threshold = 0.3):
    """
    Map a free‑form name to its primary alias via fuzzy matching (Dice/bigrams).

    :param name:    Input string (or None)
    :param lookup:  Dict mapping primary → list of aliases
    :param threshold: Minimum similarity (0–1) to accept
    :return:        The matched primary string, or None if no good match
    """
    if not name:
        return None

    q_norm = normalize_str(name)

    # Build normalized‑alias → primary map
    alias_to_primary = {}
    for primary, aliases in lookup.items():
        # also include the primary itself as an alias
        alias_to_primary[normalize_str(primary)] = primary
        for alias in aliases:
            alias_to_primary[normalize_str(alias)] = primary

    # Find best candidate by Dice score
    best_target, best_score = None, -1.0
    for norm_alias in alias_to_primary:
        score = dice_coefficient(q_norm, norm_alias)
        if score > best_score:
            best_score, best_target = score, norm_alias

    if best_score >= threshold:
        return alias_to_primary[best_target]
    return None


async def query_evidence(mp: str, disease: str | None, therapy: str | None) -> None:
    """Fetch evidence from the MCP server for the given molecular profile."""
    #logging.info("Connecting to %s …", MCP_SSE_URL)

    async with sse_client(url=MCP_SSE_URL) as (read, write), ClientSession(
        read, write
    ) as session:
        # Handshake
        #logging.info("→ initialize()")
        await session.initialize()

        # Compose argument dict (omit keys that are None)
        args: Dict[str, str] = {"molecularProfileName": mp}
        if disease:
            args["diseaseName"] = disease
        if therapy:
            args["therapyName"] = therapy

        #logging.info("→ call_tool(get_variant_evidence)")
        result = await session.call_tool(
            name="get_variant_evidence",
            arguments=args,
        )

        # Return the first content block (usually JSON text)
        if result.content:
            return result.content[0].text
        else:
            logging.warning("No content returned by the tool.")




if __name__ == "__main__":

    #For each triplet in df query the MCP server.
    #5 replicates to get a mean and std

    num_replicates = 5
    replicates = []

    t1 = time.time()

    df = pd.read_csv('./src/data/CIViC_evidence_extracts_clinical_trials_curators.csv')

    df['molecularProfile_name'].dropna()

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
    df['triplets'] = df['molecularProfile_name'] + '_' + df['disease_name'] + '_' + df['therapies']
    df['evidence_summary'] = df['evidenceType'] + '_' + df['evidenceDirection'] + '_' + df['significance']

    df["triplets"].dropna()

    num_MP = len(set(df["molecularProfile_name"]))
    
    times = []

    for i in range(num_replicates):
        for n, MP in enumerate(set(df["molecularProfile_name"])):
            time.sleep(10)
            MP = normalize_entity(MP, MP_map)
            #print(MP)
            start = time.perf_counter()
            result = asyncio.run(query_evidence(MP, None, None))
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if n % 200 == 0:
                print(n, num_MP, statistics.mean(times), 's', round((time.time()-t1) / 60, 2), 'mins')


    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0

    print('============= FINAL RESULTS =============')
    print(f"Total calls: {len(times)} (queries × replicates)")
    print(f"Mean time: {mean_time:.4f} s")
    print(f"Std dev: {stdev_time:.4f} s")

    
