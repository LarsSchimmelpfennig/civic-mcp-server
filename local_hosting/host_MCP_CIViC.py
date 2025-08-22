from mcp.server.fastmcp import FastMCP
import requests, json, sys
from difflib import SequenceMatcher
import logging
logging.basicConfig(level=logging.DEBUG)
import re
import unicodedata
from collections import Counter
import functools, sys
print = functools.partial(print, file=sys.stderr, flush=True) 
import os

mcp = FastMCP("CIViC Query App")

HEADERS = {"Content-Type": "application/json"}

with open("../src/data/disease_name_map.json", "r") as f:
    d_disease_map = json.load(f)

with open("../src/data/therapy_name_map.json", "r") as f:
    d_therapy_map = json.load(f)

with open("../src/data/molecular_profile_map.json", "r") as f:
    MP_map = json.load(f)

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

def normalize_entity(name, lookup, threshold = 0.7):
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


def _run_query(query, variables: dict) -> dict:
    print("Running GraphQL Query with variables:", variables, file=sys.stderr)
    r = requests.post(
        "https://civicdb.org/api/graphql",
        json={"query": query, "variables": variables},
        headers=HEADERS,
        timeout=30,
    )
    print("GraphQL Response:", r.text, file=sys.stderr)
    r.raise_for_status()
    return r.json()



@mcp.tool(title="CIViC evidence items for a gene variant + disease (optional) + therapy (optional). Contains clinical significance information for specific publications.")
def get_variant_evidence(molecularProfileName, diseaseName=None, therapyName=None):
    """Return CIViC JSON for the requested disease / molecular profile."""

    print(f"Tool called with variant: {molecularProfileName}, disease: {diseaseName}", file=sys.stderr, flush=True)

    molecularProfileName = normalize_entity(molecularProfileName, MP_map)
    diseaseName = normalize_entity(diseaseName, d_disease_map)

    if therapyName:
        therapyName = therapyName.split(', ')[0]

    therapyName = normalize_entity(therapyName, d_therapy_map)


    print(f"Normalized name: {molecularProfileName}", file=sys.stderr, flush=True)
    
    evidence_query = """
    query evidenceItems($molecularProfileName: String!, $diseaseName: String, $therapyName: String) {
    evidenceItems( molecularProfileName: $molecularProfileName, diseaseName: $diseaseName, therapyName: $therapyName, first: 20) {
        nodes { 
            status 
            evidenceType
            evidenceDirection 
            significance
            molecularProfile{
                variants{
                    name
                    feature{
                        name
                    }
                }
            }
            disease{
                displayName
            }
            therapies{
                name
            }
            variantOrigin 
            description
            evidenceLevel
            evidenceRating
            id 
        }
    }
    }
    """

    variables = {
        k: v
        for k, v in {
            "diseaseName": diseaseName,
            "molecularProfileName": molecularProfileName,
            "therapyName": therapyName,
        }.items()
        if v is not None and v.lower() != 'none' and v.lower() != 'null'           # keep only non-None values
    }

    resp = _run_query(evidence_query, variables)

    if 'data' not in resp: #query error
        return resp
    
    evidence_items = resp['data']['evidenceItems']['nodes']

    for item in evidence_items:
        eid = item['id']
        item['url'] = f'https://identifiers.org/civic.eid:{eid}'
        del item['id']

    return_object = {}

    return_object['Field Descriptions'] = (
        "evidenceType: Category describing the type of clinical or biological evidence (e.g., predictive, diagnostic).\n"
        "evidenceDirection: Indicates whether the evidence supports or refutes the association.\n"
        "significance: The clinical relevance of the evidence.\n"
        "description: Detailed summary of the evidence from CIViC curators.\n"
        "evidenceLevel: Describes the robustness of the study type. A - Validated association, B - Clinical evidence, C - Case study, D - Preclinical evidence, and E - Inferential association\n"
        "evidenceRating: Quality score assigned to the evidence by curators (scored 1-5).\n"
        "url: Direct link to the CIViC record for this evidence item.\n"
        "When returning information to users you MUST cite URLs used for specific information."
    )

    return_object['API Results'] = evidence_items

    return return_object


@mcp.tool(title="CIViC assertions for a gene variant + disease (optional) + therapy (optional). Contains clinical significance information across multiple publications.")
def get_variant_assertions(molecularProfileName, diseaseName=None, therapyName=None):
    """Return CIViC JSON for the requested disease / molecular profile."""

    print(f"Tool called with variant: {molecularProfileName}, disease: {diseaseName}", file=sys.stderr, flush=True)

    molecularProfileName = normalize_entity(molecularProfileName, MP_map)
    diseaseName = normalize_entity(diseaseName, d_disease_map)

    if therapyName:
        therapyName = therapyName.split(', ')[0]

    therapyName = normalize_entity(therapyName, d_therapy_map)

    print(f"Normalized name: {molecularProfileName}", file=sys.stderr, flush=True)
    
    assertions_query = """
    query assertions($molecularProfileName: String!, $diseaseName: String, $therapyName: String) {
        assertions(molecularProfileName: $molecularProfileName, diseaseName: $diseaseName, therapyName: $therapyName) {
            nodes { 
                status 
                assertionType
                assertionDirection 
                significance
                molecularProfile{
                variants{
                        name
                        feature{
                            name
                        }
                    }
                }
                disease{
                    displayName
                }
                therapies{
                    name
                } 
                summary
                id 
            }
        }
    }"""

    variables = {
        k: v
        for k, v in {
            "diseaseName": diseaseName,
            "molecularProfileName": molecularProfileName,
            "therapyName": therapyName
        }.items()
        if v is not None and v.lower() != 'none' and v.lower() != 'null'           # keep only non-None values
    }

    resp = _run_query(assertions_query, variables)

    if 'data' not in resp: #query error
        return resp
    
    assertions = resp['data']['assertions']['nodes']

    #modify to get a clickable link to CIViC
    for item in assertions:
        aid = item['id']
        item['url'] = f'https://identifiers.org/civic.aid:{aid}'
        del item['id']

    return_object = {}

    return_object['Field Descriptions'] = (
        "assertionType: Category describing the type of clinical or biological evidence (e.g., predictive, diagnostic).\n"
        "assertionDirection: Indicates whether the evidence supports or refutes the association.\n"
        "significance: The clinical relevance of the evidence.\n"
        "summary: Detailed summary of the evidence from CIViC curators.\n"
        "url: Direct link to the CIViC record for this evidence item.\n"
        "When returning information to users you MUST cite URLs used for specific information."
    )

    return_object['API Results'] = assertions

    return return_object



if __name__ == "__main__":           # <-- only runs when *you* call the file
    mcp.run(transport="stdio")       # recommended pattern in the quick-start 