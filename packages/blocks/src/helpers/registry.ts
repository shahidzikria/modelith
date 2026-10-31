// Python helper snippets (classes/functions) emitted into model.py only when used.

export interface PyHelper {
  code: string;
  deps?: string[];
  imports?: string[];
  /** Where the helper goes in model.py: building blocks (default), post-processing or losses. */
  section?: "block" | "post" | "loss";
}

export const torchHelpers: Record<string, PyHelper> = {};
export const kerasHelpers: Record<string, PyHelper> = {};

export function defineTorchHelpers(h: Record<string, PyHelper>): void {
  Object.assign(torchHelpers, h);
}

export function defineKerasHelpers(h: Record<string, PyHelper>): void {
  Object.assign(kerasHelpers, h);
}

/** Dependency-ordered, de-duplicated helper ids (deps before dependants, otherwise first-use order). */
export function resolveHelpers(ids: string[], table: Record<string, PyHelper>): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  const visit = (id: string, stack: string[]) => {
    if (seen.has(id)) return;
    const h = table[id];
    if (!h) throw new Error(`Unknown helper "${id}"`);
    if (stack.includes(id)) throw new Error(`Helper cycle: ${[...stack, id].join(' → ')}`);
    for (const d of h.deps ?? []) visit(d, [...stack, id]);
    seen.add(id);
    out.push(id);
  };
  for (const id of ids) visit(id, []);
  return out;
}
