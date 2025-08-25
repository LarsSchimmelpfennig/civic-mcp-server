from mcp.server.fastmcp import FastMCP
import requests, json, sys
from difflib import SequenceMatcher
import logging
logging.basicConfig(level=logging.DEBUG)

import functools, sys
print = functools.partial(print, file=sys.stderr, flush=True) 

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

mcp = FastMCP("CIViC Query App")

HEADERS = {"Content-Type": "application/json"}

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
def get_variant_evidence(molecularProfileName, cancerType=None, therapyName=None):
    """Return CIViC JSON for the requested disease / molecular profile."""

    #print(f"Tool called with variant: {molecularProfileName}, disease: {diseaseName}", file=sys.stderr, flush=True)

    if therapyName:
        therapyName = therapyName.split(', ')[0]

    print(f"Normalized name: {molecularProfileName}", file=sys.stderr, flush=True)
    
    evidence_query = """
    query evidenceItems($molecularProfileName: String!, $diseaseName: String, $therapyName: String) {
    evidenceItems( molecularProfileName: $molecularProfileName, diseaseName: $diseaseName, therapyName: $therapyName, first: 50) {
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
            "diseaseName": cancerType,
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
def get_variant_assertions(molecularProfileName, cancerType=None, therapyName=None):
    """Return CIViC JSON for the requested disease / molecular profile."""

    if therapyName:
        therapyName = therapyName.split(', ')[0]

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
            "diseaseName": cancerType,
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



if __name__ == "__main__":           
    mcp.run(transport="stdio")       # recommended pattern in the quick-start 