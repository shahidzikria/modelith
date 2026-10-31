// Small helpers for writing Python source from TypeScript.

/** Python literal for a JS value. Arrays become tuples, objects become dicts. */
export function py(v: unknown): string {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return v > 0 ? 'float("inf")' : 'float("-inf")';
    return String(v);
  }
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) {
    if (v.length === 1) return `(${py(v[0])},)`;
    return `(${v.map(py).join(', ')})`;
  }
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>).map(([k, x]) => `${JSON.stringify(k)}: ${py(x)}`);
    return `{${entries.join(', ')}}`;
  }
  return String(v);
}

/** Python list literal. */
export function pyList(v: unknown[]): string {
  return `[${v.map(py).join(', ')}]`;
}

/** Float literal that always contains a dot or exponent. */
export function pyFloat(n: number): string {
  const s = String(n);
  return /[.e]/.test(s) ? s : `${s}.0`;
}

/** Keyword args string, skipping undefined values. `kw({a: 1, b: undefined})` → `a=1`. */
export function kw(args: Record<string, unknown>): string {
  return Object.entries(args)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${k}=${typeof v === 'string' && v.startsWith('§') ? v.slice(1) : py(v)}`)
    .join(', ');
}

/** Mark a string as raw Python code inside `kw()`. */
export const raw = (code: string): string => `§${code}`;

export function indent(code: string, n = 4): string {
  const pad = ' '.repeat(n);
  return code
    .split('\n')
    .map((l) => (l.trim() ? pad + l : ''))
    .join('\n');
}

export function dedent(code: string): string {
  const lines = code.replace(/^\n/, '').replace(/\s+$/, '').split('\n');
  const min = Math.min(...lines.filter((l) => l.trim()).map((l) => l.match(/^ */)![0].length));
  return lines.map((l) => l.slice(min)).join('\n');
}

/** Python identifier from free text. */
export function ident(s: string): string {
  let r = s
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '');
  if (!r) r = 'x';
  if (/^[0-9]/.test(r)) r = `n_${r}`;
  if (PY_KEYWORDS.has(r)) r = `${r}_`;
  return r;
}

const PY_KEYWORDS = new Set(
  'False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self input type'.split(
    ' ',
  ),
);

/** `name(a, b, k=v)` with undefined kwargs dropped (values follow `kw()` rules). */
export function call(name: string, positional: unknown[], kwargs: Record<string, unknown> = {}): string {
  const pos = positional.map((v) => (typeof v === 'string' && v.startsWith('§') ? v.slice(1) : py(v)));
  const k = kw(kwargs);
  return `${name}(${[...pos, ...(k ? [k] : [])].join(', ')})`;
}
