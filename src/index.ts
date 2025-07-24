import gestaltSimilarity from "gestalt-pattern-matcher";

import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

// import { TfIdf, NGrams } from "natural";

import rawTherapyMap  from "./data/therapy_name_map.json";
import rawDiseaseMap  from "./data/disease_name_map.json";
//import rawGeneMap     from "./data/gene_name_map.json";
import rawMPMap from "./data/molecular_profile_map.json";

export const dTherapyMap: Record<string, string[]>  = rawTherapyMap  as Record<string, string[]>;
export const dDiseaseMap: Record<string, string[]>  = rawDiseaseMap  as Record<string, string[]>;
//export const dGeneMap:    Record<string, string[]>  = rawGeneMap     as Record<string, string[]>;
export const dMPMap:    Record<string, string[]>  = rawMPMap     as Record<string, string[]>;

// import diseaseData from "./data/inverted_resolver_disease_500.json";
// import therapyData from "./data/inverted_resolver_therapy_500.json";
// import molecularData from "./data/inverted_resolver_molecular_500.json";

export interface ResolverData {
  vocabulary: Record<string, number>;
  idf: number[];
  alias_list: string[];
  alias_to_key: Record<string, string>;
  index: Record<string, number[][]>;
  threshold: number;
}

// helper to extract 3‑grams
function charNgrams(s: string, n = 3): string[] {
  const out: string[] = [];
  s = s.toLowerCase();
  for (let i = 0; i + n <= s.length; i++) out.push(s.slice(i, i + n));
  return out;
}

/**
 * Generic cosine‑TFIDF resolver.
 * Pass in the mention and whichever resolverData you want.
 */
export function resolve(
  mention: string | null | undefined,
  {
    vocabulary,
    idf,
    alias_list,
    alias_to_key,
    index,
    threshold,
  }: ResolverData
): string | null {
  if (!mention) return null;

  // 1) build term‑counts per feature index
  const counts: Record<number, number> = {};
  for (const gram of charNgrams(mention)) {
    const idx = vocabulary[gram];
    if (idx != null) counts[idx] = (counts[idx] || 0) + 1;
  }

  // 2) TF*IDF + L2‑normalize
  const vec: Record<number, number> = {};
  let norm2 = 0;
  for (const [idxStr, tf] of Object.entries(counts)) {
    const idx = +idxStr;
    const w = tf * idf[idx];
    vec[idx] = w;
    norm2 += w * w;
  }
  const norm = Math.sqrt(norm2);
  if (norm === 0) return null;
  for (const k of Object.keys(vec)) {
    vec[+k] /= norm;
  }

  // 3) accumulate scores via inverted index
  const scores = new Float32Array(alias_list.length);
  for (const [featIdxStr, w] of Object.entries(vec)) {
    const featIdx = +featIdxStr;
    // find the 3‑gram that maps to this feature index
    const gram = Object.keys(vocabulary).find(
      (g) => vocabulary[g] === featIdx
    )!;
    const postings = index[gram] || [];
    for (const [aliasIdx, weight] of postings) {
      scores[aliasIdx] += weight * w;
    }
  }

  // 4) pick best
  let bestScore = -Infinity;
  let bestIdx = -1;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > bestScore) {
      bestScore = scores[i];
      bestIdx = i;
    }
  }
  if (bestScore < threshold) return null;

  const bestAlias = alias_list[bestIdx];
  return alias_to_key[bestAlias] || null;
}

// convenience wrappers
// export const resolveDisease = (mention: string | null | undefined) =>
//   resolve(mention, diseaseData as ResolverData);

// export const resolveTherapy = (mention: string | null | undefined) =>
//   resolve(mention, therapyData as ResolverData);

// export const resolveMolecularProfile = (mention: string | null | undefined) =>
//   resolve(mention, molecularData as ResolverData);


export function normalizeEntity(
  name: string | undefined | null,
  lookup: Record<string, string[]>,
  threshold = 0.7,
  returnInput = false,
): string | null {
  if (!name) return null;

  const nameLower = name.toLowerCase();
  let best: string | null = null;
  let bestScore = 0;

  for (const [primary, synonyms] of Object.entries(lookup)) {
    for (const cand of [primary, ...synonyms]) {
      const ratio = gestaltSimilarity(nameLower, cand.toLowerCase());
      if (ratio > bestScore) {
        bestScore = ratio;
        best = primary;
      }
    }
  }
  return bestScore >= threshold ? best : returnInput ? name : null;
}

/** -----------------------------------------------------------
 *  Helper: remove null / undefined properties from an object
 *  --------------------------------------------------------- */
function compact<T extends Record<string, unknown>>(obj: T): Partial<T> {
  return Object.fromEntries(
    Object.entries(obj).filter(([, v]) => v != null),
  ) as Partial<T>;          // ← assert the narrowed shape
}

// ========================================
// API CONFIGURATION - Customize for your GraphQL API
// ========================================

export const API_CONFIG = {
  name:        "CivicExplorer",
  version:     "0.1.0",
  description: "Fixed‑schema MCP tools for the CIViC GraphQL API",
  mcpSpecVersion: "2025-06-18",
  features: {
    structuredToolOutput: true,
    metaFields:           true,
    protocolVersionHeaders: true,
    titleFields:          true,
    toolAnnotations:      true,
  },
  headers: {
    Accept:      "application/vnd.civicdb.v2+json",
    "User-Agent": "MCPCivicServer/0.1.0",
    "Content-Type": "application/json",
  },
} as const;

/** -----------------------------------------------------------------
 *  Tool definitions
 *  ---------------------------------------------------------------- */
type EvidenceInput   = { molecularProfileName: string; diseaseName?: string; therapyName?: string };
type AssertionsInput = { molecularProfileName: string; diseaseName?: string };

export const tools = {
  /** ───────────────────────── get_variant_evidence ─────────────────────── */
  getVariantEvidence: {
    name: "get_variant_evidence",
    description:
      "Return up to 10 evidence items for a CIViC molecular profile, " +
      "optionally filtered by disease and/or therapy. Cite URLs used for specific information.",
    inputSchema: {
      molecularProfileName: z.string(),
      diseaseName:          z.string().optional(),
      therapyName:          z.string().optional(),
    },
    annotations: {
      destructive: false,
      idempotent:  true,
      cacheable:   false,
      world_interaction: "open",
      side_effects: ["external_api_calls"],
      resource_usage: "network_io_heavy",
    },
    async handler({ molecularProfileName, diseaseName, therapyName }: EvidenceInput) {

      // const variables = compact({
      //   molecularProfileName: resolveMolecularProfile(molecularProfileName),
      //   diseaseName:          resolveDisease(diseaseName),
      //   therapyName:          resolveTherapy(therapyName),
      // });

      const variables = compact({
        molecularProfileName: normalizeEntity(molecularProfileName, dMPMap,   0.7, true),
        diseaseName:          normalizeEntity(diseaseName,      dDiseaseMap, 0.7),
        therapyName:          normalizeEntity(therapyName,      dTherapyMap, 0.7),
      });

      const query = /* GraphQL */ `
        query evidenceItems($molecularProfileName: String!, $diseaseName: String, $therapyName: String) {
        evidenceItems( molecularProfileName: $molecularProfileName, diseaseName: $diseaseName, therapyName: $therapyName, first: 10) {
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
                source {
                  sourceUrl
                }
                id 
            }
        }
      }`;

        const res = await fetch("https://civicdb.org/api/graphql", {
          method: "POST",
          headers: API_CONFIG.headers,
          body: JSON.stringify({ query, variables }),
        }).then((r) => r.json()) as {
          data?: { evidenceItems?: { nodes: Record<string, unknown>[] } };
          errors?: unknown[];
        };

        // Default fallback if query failed
        const rawItems = res.data?.evidenceItems?.nodes ?? res;

        // Only transform if it's an array of evidence items
        const evidenceItems = Array.isArray(rawItems)
          ? rawItems.map((item) => {
              const copy = { ...item } as Record<string, unknown>;
              const id = copy["id"];
              if (typeof id === "string" || typeof id === "number") {
                copy["url"] = `https://identifiers.org/civic.eid:${id}`;
              }
              delete copy["id"];
              return copy;
            })
          : rawItems;

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(evidenceItems, null, 2),
            },
          ],
          _meta: {
            row_count: Array.isArray(evidenceItems) ? evidenceItems.length : undefined,
          },
        };
    },
  },

  /** ───────────────────────── get_variant_assertions ───────────────────── */
  getVariantAssertions: {
    name: "get_variant_assertions",
    description:
      "Return CIViC assertions for a molecular profile; optionally filter by disease. Cite URLs used for specific information.",
    inputSchema: {
      molecularProfileName: z.string(),
      diseaseName:          z.string().optional(),
    },
    annotations: {
      destructive: false,
      idempotent:  true,
      cacheable:   false,
      world_interaction: "open",
      side_effects: ["external_api_calls"],
      resource_usage: "network_io_heavy",
    },
    async handler({ molecularProfileName, diseaseName }: AssertionsInput) {

      // const variables = compact({
      //   molecularProfileName: resolveMolecularProfile(molecularProfileName),
      //   diseaseName:          resolveDisease(diseaseName),
      // });

      const variables = compact({
        molecularProfileName: normalizeEntity(molecularProfileName, dMPMap,   0.7, true),
        diseaseName:          normalizeEntity(diseaseName,      dDiseaseMap, 0.7)
      });

      const query = /* GraphQL */ `
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
      }`;

        const res = await fetch("https://civicdb.org/api/graphql", {
          method: "POST",
          headers: API_CONFIG.headers,
          body: JSON.stringify({ query, variables }),
        }).then((r) => r.json()) as {
          data?: { assertions?: { nodes: Record<string, unknown>[] } };
          errors?: unknown[];
        };

        const rawItems = res.data?.assertions?.nodes ?? res;

        // Only transform if it's an array of assertions
        const assertions = Array.isArray(rawItems)
          ? rawItems.map((item) => {
              const copy = { ...item } as Record<string, unknown>;
              const id = copy["id"];
              if (typeof id === "string" || typeof id === "number") {
                copy["url"] = `https://identifiers.org/civic.aid:${id}`;
              }
              delete copy["id"];
              return copy;
            })
          : rawItems;

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(assertions, null, 2),
            },
          ],
          _meta: {
            row_count: Array.isArray(assertions) ? assertions.length : undefined,
          },
        };
    },
  },
} as const;



// -------------------------------------------------------------
// MCP SERVER (only the two fixed tools)
// -------------------------------------------------------------
class CivicMCP extends McpAgent {
  server = new McpServer({
    name:        API_CONFIG.name,
    version:     API_CONFIG.version,
    description: API_CONFIG.description,},
    
    {
    instructions: `
Use the tools to answer oncology variant questions for the Clinical Interpretations of Variants in Cancer (CIViC) knowledgebase.

**Definition of each evidence type**
• Diagnostic – Evidence pertains to a variant’s impact on patient diagnosis (cancer subtype).  
• Predictive – Evidence pertains to a variant’s effect on therapeutic response.  
• Prognostic – Evidence pertains to a variant’s impact on disease progression, severity, or patient survival.  
• Predisposing – Evidence pertains to a germline molecular profile’s role in conferring susceptibility to disease (including pathogenicity evaluations).  
• Oncogenic – Evidence pertains to a somatic variant’s involvement in tumor pathogenesis as described by the Hallmarks of Cancer.  
• Functional – Evidence pertains to a variant that alters biological function from the reference state.

Always call **get_variant_evidence** and **get_variant_assertions** to determine clinical significance.
Gene names are normalized to how they appear in CIViC. If the gene name in CIViC descriptions/summaries does not match the input name assume it is an alias. 
IMPORTANT: When using information from a specific evidence item or assertion, cite it with the associated url.
    `,
  });

  async init() {
    /* register fixed-schema tools */
    const { getVariantEvidence, getVariantAssertions } = tools;

    this.server.tool(
      getVariantEvidence.name,
      getVariantEvidence.description,
      getVariantEvidence.inputSchema,
      getVariantEvidence.handler,
    );

    this.server.tool(
      getVariantAssertions.name,
      getVariantAssertions.description,
      getVariantAssertions.inputSchema,
      getVariantAssertions.handler,
    );
  }
}

// -------------------------------------------------------------
// CLOUDFLARE WORKER RUNTIME (SSE only)
// -------------------------------------------------------------
interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}


export default {
  async fetch(
    request: Request,
    env: Env,               // keep the typed Env like before
    ctx: ExecutionContext
  ): Promise<Response> {
    const url = new URL(request.url);

    console.log(Object.keys(env));

    /* ────────────────────────────────────────────────
       SSE transport (Claude Desktop, Cursor, etc.)
    ─────────────────────────────────────────────────*/
    if (url.pathname === "/sse" || url.pathname.startsWith("/sse/")) {
      // MCP 2025-06-18: client may send its protocol version
      const protocolVersion = request.headers.get("MCP-Protocol-Version");

      // @ts-ignore – serveSSE helper is mixed-in by CivicMCP
      const response = await CivicMCP.serveSSE("/sse").fetch(request, env, ctx);

      // Mirror the header back so the client sees what the server supports
      if (protocolVersion && response instanceof Response) {
        const headers = new Headers(response.headers);
        headers.set("MCP-Protocol-Version", protocolVersion);
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers
        });
      }

      return response; // unchanged fallback
    }

    return new Response(
      `${API_CONFIG.name} – MCP Server ${API_CONFIG.version}. Use /sse for MCP transport.`,
      { status: 200, headers: { "Content-Type": "text/plain" } }
    );
  }
};

// Export the class for tests if you like
export { CivicMCP };