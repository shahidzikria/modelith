// Small helpers for writing Python source from TypeScript.

/** Python literal for a JS value. Arrays become tuples (nested arrays too). */
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

/** Float literal that always contains a dot or exponent. */
export function pyFloat(n: number): string {
  const s = String(n);
  return /[.e]|inf/.test(s) ? s : `${s}.0`;
}

/** Python identifier from free text. */
export function ident(s: string): string {
  let r = s
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '');
  if (!r) r = 'x';
  if (/^[0-9]/.test(r)) r = `n_${r}`;
  return r;
}

export function indent(code: string, n = 4): string {
  const pad = ' '.repeat(n);
  return code
    .split('\n')
    .map((l) => (l.trim() ? pad + l : ''))
    .join('\n');
}

/** Strip the common leading indentation (lets emitters use template literals naturally). */
export function dedent(code: string): string {
  const lines = code.replace(/^\n/, '').replace(/\s+$/, '').split('\n');
  const min = Math.min(...lines.filter((l) => l.trim()).map((l) => l.match(/^ */)![0].length));
  return lines.map((l) => l.slice(min)).join('\n');
}

/** '#rrggbb' → [r, g, b]. */
export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex) ?? ['', '00', '00', '00'];
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

/** Raw Python code for a generated constant (keeps e.g. `cv2.INTER_AREA` unquoted). */
export const pyRaw = (code: string) => ({ __raw: code });
