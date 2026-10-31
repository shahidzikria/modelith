import type { BlockDef } from './types';

const blocks = new Map<string, BlockDef>();

/** Register a block. Block files call this at import time; `src/blocks/index.ts` imports them all. */
export function defineBlock(def: BlockDef): BlockDef {
  if (blocks.has(def.type)) throw new Error(`Duplicate block type: ${def.type}`);
  blocks.set(def.type, def);
  return def;
}

export function getBlock(type: string): BlockDef | undefined {
  return blocks.get(type);
}

export function requireBlock(type: string): BlockDef {
  const b = blocks.get(type);
  if (!b) throw new Error(`Unknown block type: ${type}`);
  return b;
}

export function allBlocks(): BlockDef[] {
  return [...blocks.values()];
}

/** Default param values for a block type. */
export function defaultParams(type: string): Record<string, unknown> {
  const def = requireBlock(type);
  return Object.fromEntries(def.params.map((p) => [p.key, structuredClone(p.default)]));
}
