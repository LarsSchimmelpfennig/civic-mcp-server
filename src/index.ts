import gestaltSimilarity from "gestalt-pattern-matcher";

import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

import rawTherapyMap  from "./data/therapy_name_map.json";
import rawDiseaseMap  from "./data/disease_name_map.json";
import rawGeneMap     from "./data/gene_name_map.json";

export const dTherapyMap: Record<string, string[]>  = rawTherapyMap  as Record<string, string[]>;
export const dDiseaseMap: Record<string, string[]>  = rawDiseaseMap  as Record<string, string[]>;
export const dGeneMap:    Record<string, string[]>  = rawGeneMap     as Record<string, string[]>;


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
      "optionally filtered by disease and/or therapy.",
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
        molecularProfileName: normalizeEntity(molecularProfileName, dGeneMap,   0.7, true),
        diseaseName:          normalizeEntity(diseaseName,      dDiseaseMap, 0.7),
        therapyName:          normalizeEntity(therapyName,      dTherapyMap, 0.7),
      });

      const query = /* GraphQL */ `
        query EvidenceItems(
          $molecularProfileName: String!
          $diseaseName: String
          $therapyName: String
        ) {
          evidenceItems(
            molecularProfileName: $molecularProfileName
            diseaseName:          $diseaseName
            therapyName:          $therapyName
            first: 10
          ) {
            nodes {
              status
              evidenceDirection
              significance
              disease   { displayName }
              therapies { name }
              variantOrigin
              description
              evidenceLevel
              evidenceRating
            }
          }
        }`;

        const res = await fetch("https://civicdb.org/api/graphql", { method: "POST",
        headers: API_CONFIG.headers,
        body:   JSON.stringify({ query, variables }), }).then(r => r.json()) as {
                data?: { evidenceItems?: { nodes: unknown[] } };
                errors?: unknown[];
        };
        
        const payload = res.data?.evidenceItems?.nodes ?? res;   // whatever you want to surface

        return {
                content: [
                        {type: "text" as const, text: JSON.stringify(payload, null, 2),},
                ],
                _meta: {
                row_count: Array.isArray(payload) ? payload.length : undefined,
                // add any extra metadata you like
                },
        };
    },
  },

  /** ───────────────────────── get_variant_assertions ───────────────────── */
  getVariantAssertions: {
    name: "get_variant_assertions",
    description:
      "Return CIViC assertions for a molecular profile; optionally filter by disease.",
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
      const variables = compact({
        molecularProfileName: normalizeEntity(molecularProfileName, dGeneMap,   0.7, true),
        diseaseName:          normalizeEntity(diseaseName,      dDiseaseMap, 0.7),
      });

      const query = /* GraphQL */ `
        query Assertions(
          $molecularProfileName: String!
          $diseaseName: String
        ) {
          assertions(
            molecularProfileName: $molecularProfileName
            diseaseName:          $diseaseName
          ) {
            nodes {
              status
              assertionDirection
              significance
              summary
            }
          }
        }`;
        const res = await fetch("https://civicdb.org/api/graphql", { method: "POST",
        headers: API_CONFIG.headers,
        body:   JSON.stringify({ query, variables }), 
        }).then(r => r.json()) as {
                data?: { assertions?: { nodes: unknown[] } };
                errors?: unknown[];
        };
        const payload = res.data?.assertions?.nodes ?? res;

        return {
                content: [
        {
                type: "text" as const,
                text: JSON.stringify(payload, null, 2),
        },],
        _meta: {
                row_count: Array.isArray(payload) ? payload.length : undefined,},
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