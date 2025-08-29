import gestaltSimilarity from "gestalt-pattern-matcher";

import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

// import { TfIdf, NGrams } from "natural";

import rawTherapyMap  from "./data/therapy_name_map.json";
import rawDiseaseMap  from "./data/disease_name_map.json";
import rawMPMap from "./data/molecular_profile_map.json";

//import rawGeneMap     from "./data/gene_name_map.json";

export const dTherapyMap: Record<string, string[]>  = rawTherapyMap  as Record<string, string[]>;
export const dDiseaseMap: Record<string, string[]>  = rawDiseaseMap  as Record<string, string[]>;
export const dMPMap:    Record<string, string[]>  = rawMPMap     as Record<string, string[]>;

//export const dGeneMap:    Record<string, string[]>  = rawGeneMap     as Record<string, string[]>;

// import diseaseData from "./data/inverted_resolver_disease_500.json";
// import therapyData from "./data/inverted_resolver_therapy_500.json";
// import molecularData from "./data/inverted_resolver_molecular_500.json";

import { findBestMatch } from 'string-similarity'

function normalizeStr(s: string): string {
  return s
    .normalize('NFKD')                   // decompose accents
    .replace(/[\u0300-\u036f]/g, '')     // strip diacritics
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')        // keep only letters, digits, spaces
    .replace(/\s+/g, ' ')                // collapse runs of spaces
    .trim()
}

/**
 * Map a free‑form name to its primary alias via fuzzy matching.
 *
 * @param name      Input string (may be undefined or null)
 * @param lookup    Record<primary, aliases[]>
 * @param threshold Minimum similarity (0–1) to accept a match
 * @returns         The matched primary string, or undefined if below threshold or name missing
 */
export function normalizeEntity(
  name: string | undefined | null,
  lookup: Record<string, string[]>,
  threshold: number = 0.7
): string | undefined {
  if (!name) {
    return undefined
  }

  const qNorm = normalizeStr(name)

  // Build a map from normalized‑alias → primary
  const aliasToPrimary: Record<string, string> = {}
  for (const [primary, aliases] of Object.entries(lookup)) {
    aliasToPrimary[normalizeStr(primary)] = primary
    for (const alias of aliases) {
      aliasToPrimary[normalizeStr(alias)] = primary
    }
  }

  const candidates = Object.keys(aliasToPrimary)
  const { bestMatch } = findBestMatch(qNorm, candidates)

  return bestMatch.rating >= threshold
    ? aliasToPrimary[bestMatch.target]
    : undefined
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

export const tools = {
  /** ───────────────────────── get_variant_evidence ─────────────────────── */
  getVariantEvidence: {
    name: "get_variant_evidence",
    description:
      "Retrieves evidence items for a CIViC molecular profile, " +
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
        molecularProfileName: normalizeEntity(molecularProfileName, dMPMap,   0.7),
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

        const instructions =
          "evidenceType: Category describing the type of clinical or biological evidence (e.g., predictive, diagnostic).\n" +
          "evidenceDirection: Indicates whether the evidence supports or refutes the association.\n" +
          "significance: The clinical relevance of the evidence.\n" +
          "description: Detailed summary of the evidence from CIViC curators.\n" +
          "evidenceLevel: Describes the robustness of the study type. A - Validated association, B - Clinical evidence, C - Case study, D - Preclinical evidence, and E - Inferential association\n" +
          "evidenceRating: Quality score assigned to the evidence by curators (scored 1-5).\n" +
          "url: Direct link to the CIViC record for this evidence item.\n" +
          "When returning information to users you MUST cite URLs used for specific information.";

        const payload = {
          instructions,
          "API Results": evidenceItems,
        };

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(payload, null, 2),
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
      "Retrieves CIViC assertions for a molecular profile; optionally filter by disease. Cite URLs used for specific information.",
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

      const variables = compact({
        molecularProfileName: normalizeEntity(molecularProfileName, dMPMap,   0.7),
        diseaseName:          normalizeEntity(diseaseName,      dDiseaseMap, 0.7),
        therapyName:          normalizeEntity(therapyName,      dTherapyMap, 0.7),
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

        const instructions =
          "assertionType: Category describing the type of clinical or biological evidence (e.g., predictive, diagnostic).\n" +
          "assertionDirection: Indicates whether the evidence supports or refutes the association.\n" +
          "significance: The clinical relevance of the evidence.\n" +
          "summary: Detailed summary of the evidence from CIViC curators.\n" +
          "url: Direct link to the CIViC record for this evidence item.\n" +
          "When returning information to users you MUST cite URLs used for specific information.";

        const payload = {
          instructions,
          "API Results": assertions,
        };

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(payload, null, 2),
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
      Use the tools to answer precision oncology variant questions for the Clinical Interpretations of Variants in Cancer (CIViC) knowledgebase.`,
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


// export default {
//   async fetch(
//     request: Request,
//     env: Env,               // keep the typed Env like before
//     ctx: ExecutionContext
//   ): Promise<Response> {
//     const url = new URL(request.url);

//     console.log(Object.keys(env));

//     /* ────────────────────────────────────────────────
//        SSE transport (Claude Desktop, Cursor, etc.)
//     ─────────────────────────────────────────────────*/
//     if (url.pathname === "/sse" || url.pathname.startsWith("/sse/")) {
//       // MCP 2025-06-18: client may send its protocol version
//       const protocolVersion = request.headers.get("MCP-Protocol-Version");

//       // @ts-ignore – serveSSE helper is mixed-in by CivicMCP
//       const response = await CivicMCP.serveSSE("/sse").fetch(request, env, ctx);

//       // Mirror the header back so the client sees what the server supports
//       if (protocolVersion && response instanceof Response) {
//         const headers = new Headers(response.headers);
//         headers.set("MCP-Protocol-Version", protocolVersion);
//         return new Response(response.body, {
//           status: response.status,
//           statusText: response.statusText,
//           headers
//         });
//       }

//       return response; // unchanged fallback
//     }

//     return new Response(
//       `${API_CONFIG.name} – MCP Server ${API_CONFIG.version}. Use /sse for MCP transport.`,
//       { status: 200, headers: { "Content-Type": "text/plain" } }
//     );
//   }
// };

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext
  ): Promise<Response> {
    const url = new URL(request.url);

    console.log(Object.keys(env));

    /* ────────────────────────────────────────────────
       NEW: Streamable HTTP transport (/mcp)
    ─────────────────────────────────────────────────*/
    if (url.pathname === "/mcp" || url.pathname.startsWith("/mcp/")) {
      const protocolVersion = request.headers.get("MCP-Protocol-Version");

      // @ts-ignore – serve helper is mixed-in by CivicMCP (Streamable HTTP)
      const response = await CivicMCP.serve("/mcp").fetch(request, env, ctx);

      if (protocolVersion && response instanceof Response) {
        const headers = new Headers(response.headers);
        headers.set("MCP-Protocol-Version", protocolVersion);
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers
        });
      }
      return response;
    }

    /* ────────────────────────────────────────────────
       Legacy SSE transport (kept for now)
    ─────────────────────────────────────────────────*/
    if (url.pathname === "/sse" || url.pathname.startsWith("/sse/")) {
      const protocolVersion = request.headers.get("MCP-Protocol-Version");

      // @ts-ignore – serveSSE helper is mixed-in by CivicMCP (SSE)
      const response = await CivicMCP.serveSSE("/sse").fetch(request, env, ctx);

      if (protocolVersion && response instanceof Response) {
        const headers = new Headers(response.headers);
        headers.set("MCP-Protocol-Version", protocolVersion);
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers
        });
      }
      return response;
    }

    return new Response(
      `${API_CONFIG.name} – MCP Server ${API_CONFIG.version}. Use /mcp (Streamable HTTP) or /sse (legacy).`,
      { status: 200, headers: { "Content-Type": "text/plain" } }
    );
  }
};

// Export the class for tests if you like
export { CivicMCP };