import gestaltSimilarity from "gestalt-pattern-matcher";

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


function main() {
  /* -------- inspect the maps -------- */
  console.log("\n=== therapy map sample ===");
  Object.entries(dTherapyMap)
        .slice(0, 5)               // first 5 so you don’t drown in output
        .forEach(([k, v]) => console.log(k, "⇒", v));

  /* -------- normalizeEntity checks -------- */
  const tests = [
    ["erlotinib", dTherapyMap],
    ["NS7",     dGeneMap],
    ["glioblastoma", dDiseaseMap],
  ] as const;

  for (const [q, map] of tests) {
    const hit = normalizeEntity(q, map, 0.7, /*returnInput*/ true);
    console.log(`normalizeEntity("${q}")  →  ${hit}`);
  }

  /* -------- compact checks -------- */
  const obj = {
    a: 1,
    b: null,
    c: undefined,
    d: "kept",
  };
  console.log("\ncompact test →", compact(obj));
}

main();
