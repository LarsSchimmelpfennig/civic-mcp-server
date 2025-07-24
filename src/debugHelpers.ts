// import gestaltSimilarity from "gestalt-pattern-matcher";
// import { TfIdf, NGrams } from "natural";

// import rawTherapyMap  from "./data/therapy_name_map.json";
// import rawDiseaseMap  from "./data/disease_name_map.json";
// import rawGeneMap     from "./data/gene_name_map.json";
// import rawMPMap from "./data/molecular_profile_map.json";

// export const dTherapyMap: Record<string, string[]>  = rawTherapyMap  as Record<string, string[]>;
// export const dDiseaseMap: Record<string, string[]>  = rawDiseaseMap  as Record<string, string[]>;
// export const dGeneMap:    Record<string, string[]>  = rawGeneMap     as Record<string, string[]>;
// export const dMPMap:    Record<string, string[]>  = rawMPMap     as Record<string, string[]>;

// resolver.ts
import diseaseData from "./data/inverted_resolver_disease_500.json";
import therapyData from "./data/inverted_resolver_therapy_500.json";
import molecularData from "./data/inverted_resolver_molecular_500.json";

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
export const resolveDisease = (mention: string | null | undefined) =>
  resolve(mention, diseaseData as ResolverData);

export const resolveTherapy = (mention: string | null | undefined) =>
  resolve(mention, therapyData as ResolverData);

export const resolveMolecularProfile = (mention: string | null | undefined) =>
  resolve(mention, molecularData as ResolverData);

// export function buildResolver(
//   mpMap: Record<string, string[]>,
//   threshold = 0.25
// ): (mention: string | null | undefined) => string | null {
//   // 1. Flatten aliases and reverse lookup
//   const aliasList: string[] = [];
//   const aliasToKey: Record<string, string> = {};
//   for (const key in mpMap) {
//     for (const alias of mpMap[key]) {
//       const lower = alias.toLowerCase();
//       aliasList.push(lower);
//       aliasToKey[lower] = key;
//     }
//   }

//   const N = aliasList.length;

//   // 2. Build char-3-gram lists for each alias
//   const aliasGrams: string[][] = aliasList.map(a => {
//     const grams: string[] = [];
//     for (let i = 0; i + 3 <= a.length; i++) {
//       grams.push(a.substring(i, i + 3));
//     }
//     return grams;
//   });

//   // 3. Compute document frequency (DF) and IDF
//   const df: Record<string, number> = {};
//   aliasGrams.forEach(grams => {
//     new Set(grams).forEach(g => {
//       df[g] = (df[g] || 0) + 1;
//     });
//   });
//   const idf: Record<string, number> = {};
//   for (const gram in df) {
//     idf[gram] = Math.log(N / df[gram]);
//   }

//   // 4. Precompute L2-normalized TF–IDF vectors for each alias
//   const aliasVecs: Array<Map<string, number>> = aliasGrams.map(grams => {
//     // term frequencies in this alias
//     const tf: Record<string, number> = {};
//     grams.forEach(g => { tf[g] = (tf[g] || 0) + 1; });

//     // raw TF–IDF
//     const vec = new Map<string, number>();
//     let sumSq = 0;
//     for (const g in tf) {
//       const w = tf[g] * (idf[g] ?? 0);
//       vec.set(g, w);
//       sumSq += w * w;
//     }

//     // normalize
//     const norm = Math.sqrt(sumSq) || 1;
//     for (const [g, w] of vec) {
//       vec.set(g, w / norm);
//     }
//     return vec;
//   });

//   // 5. The resolver function
//   return function resolve(mention: string | null | undefined): string | null {
//     if (!mention) return mention ?? null;

//     const m = mention.toLowerCase();
//     // build char-3-grams for the mention
//     const mGrams: string[] = [];
//     for (let i = 0; i + 3 <= m.length; i++) {
//       mGrams.push(m.substring(i, i + 3));
//     }
//     // term frequencies
//     const tf: Record<string, number> = {};
//     mGrams.forEach(g => { tf[g] = (tf[g] || 0) + 1; });

//     // raw TF–IDF for the mention
//     const mVec = new Map<string, number>();
//     let sumSq = 0;
//     for (const g in tf) {
//       if (idf[g] === undefined) continue;
//       const w = tf[g] * idf[g];
//       mVec.set(g, w);
//       sumSq += w * w;
//     }
//     // normalize
//     const norm = Math.sqrt(sumSq) || 1;
//     for (const [g, w] of mVec) {
//       mVec.set(g, w / norm);
//     }

//     // compute cosine similarity against each alias
//     let bestScore = -Infinity;
//     let bestIdx = -1;
//     for (let i = 0; i < aliasList.length; i++) {
//       const vec = aliasVecs[i];
//       let dot = 0;
//       for (const [g, mw] of mVec) {
//         const aw = vec.get(g);
//         if (aw !== undefined) dot += aw * mw;
//       }
//       if (dot > bestScore) {
//         bestScore = dot;
//         bestIdx = i;
//       }
//     }

//     if (bestScore < threshold) return null;
//     const bestAlias = aliasList[bestIdx];
//     return aliasToKey[bestAlias] ?? null;
//   };
// }

// export const MPResolver = buildResolver(dMPMap, 0.3);
// export const diseaseResolver = buildResolver(dDiseaseMap, 0.3);
// export const therapyResolver = buildResolver(dTherapyMap, 0.3);

// export function normalizeEntity(
//   name: string | undefined | null,
//   lookup: Record<string, string[]>,
//   threshold = 0.7,
//   returnInput = false,
// ): string | null {
//   if (!name) return null;

//   const nameLower = name.toLowerCase();
//   let best: string | null = null;
//   let bestScore = 0;

//   for (const [primary, synonyms] of Object.entries(lookup)) {
//     for (const cand of [primary, ...synonyms]) {
//       const ratio = gestaltSimilarity(nameLower, cand.toLowerCase());
//       if (ratio > bestScore) {
//         bestScore = ratio;
//         best = primary;
//       }
//     }
//   }
//   return bestScore >= threshold ? best : returnInput ? name : null;
// }

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
  // console.log("\n=== therapy map sample ===");
  // Object.entries(dTherapyMap)
  //       .slice(0, 5)               // first 5 so you don’t drown in output
  //       .forEach(([k, v]) => console.log(k, "⇒", v));

  console.log(resolveTherapy('erlotib'))
  console.log(resolveMolecularProfile('EFR::P1A'))
  console.log(resolveDisease('Testicular Cancer'))

  // for (const [q, map] of tests) {
  //   const hit = MPResolver(q, map, 0.7, /*returnInput*/ true);
  //   console.log(`normalizeEntity("${q}")  →  ${hit}`);
  // }

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
