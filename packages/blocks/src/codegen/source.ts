// Line-based source builder that remembers which graph node produced each line
// (two-way code ↔ graph linking) and wraps over-long calls Black-style.

export const MAX_COLS = 100;

export interface GeneratedFile {
  path: string;
  content: string;
  /** 1-based line number → node id. */
  lineMap?: Record<number, string>;
  language: 'python' | 'yaml' | 'text' | 'markdown' | 'json';
}

export class Source {
  private lines: string[] = [];
  private nodes: (string | undefined)[] = [];

  /** Append text (may contain newlines) at an indent level, tagging every line with a node id. */
  add(text: string, indentLevel = 0, nodeId?: string): this {
    const pad = '    '.repeat(indentLevel);
    for (const raw of text.split('\n')) {
      const line = raw.trim() ? pad + raw : '';
      for (const l of wrapLine(line)) {
        this.lines.push(l);
        this.nodes.push(nodeId);
      }
    }
    return this;
  }

  /** Like add(), but only the first line is indented; continuation lines are already absolute. */
  addCont(text: string, indentLevel: number, nodeId?: string): this {
    const [first, ...rest] = text.split('\n');
    this.add(first, indentLevel, nodeId);
    for (const l of rest) this.add(l, 0, nodeId);
    return this;
  }

  blank(n = 1): this {
    for (let i = 0; i < n; i++) {
      this.lines.push('');
      this.nodes.push(undefined);
    }
    return this;
  }

  file(path: string, language: GeneratedFile['language'] = 'python'): GeneratedFile {
    // Collapse 3+ blank lines into 2 and strip trailing blanks.
    const outLines: string[] = [];
    const outNodes: (string | undefined)[] = [];
    let blanks = 0;
    this.lines.forEach((l, i) => {
      if (!l.trim()) {
        blanks++;
        if (blanks > 2) return;
      } else blanks = 0;
      outLines.push(l.replace(/\s+$/, ''));
      outNodes.push(this.nodes[i]);
    });
    while (outLines.length && !outLines[outLines.length - 1]) {
      outLines.pop();
      outNodes.pop();
    }
    const lineMap: Record<number, string> = {};
    outNodes.forEach((n, i) => {
      if (n) lineMap[i + 1] = n;
    });
    return { path, content: outLines.join('\n') + '\n', lineMap, language };
  }
}

/** Split `a, b(c, d), "e,f"` on top-level commas. */
export function splitTopLevel(s: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quote: string | null = null;
  let cur = '';
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote) {
      cur += ch;
      if (ch === '\\') {
        cur += s[++i] ?? '';
      } else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") quote = ch;
    else if ('([{'.includes(ch)) depth++;
    else if (')]}'.includes(ch)) depth--;
    if (ch === ',' && depth === 0) {
      parts.push(cur.trim());
      cur = '';
    } else cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());
  return parts;
}

/** Wrap a line longer than MAX_COLS whose tail is a call / literal: `head(args)` → one arg per line. */
export function wrapLine(line: string): string[] {
  if (line.length <= MAX_COLS || line.trimStart().startsWith('#') || /"""/.test(line)) return [line];
  const tail = line.endsWith(':') ? ':' : '';
  const body = tail ? line.slice(0, -1) : line;
  const closer = body[body.length - 1];
  const openerFor: Record<string, string> = { ')': '(', ']': '[', '}': '{' };
  if (!openerFor[closer]) return [line];
  // Find the opening bracket matching the final closing one.
  let depth = 0;
  let quote: string | null = null;
  let open = -1;
  for (let i = body.length - 1; i >= 0; i--) {
    const ch = body[i];
    if (quote) {
      if (ch === quote && body[i - 1] !== '\\') quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") quote = ch;
    else if (')]}'.includes(ch)) depth++;
    else if ('([{'.includes(ch)) {
      depth--;
      if (depth === 0) {
        open = i;
        break;
      }
    }
  }
  if (open < 0) return [line];
  const head = body.slice(0, open + 1);
  const inner = body.slice(open + 1, -1);
  const args = splitTopLevel(inner);
  if (args.length === 0) return [line];
  const pad = line.match(/^ */)![0];
  const argPad = pad + '    ';
  const out = [head];
  for (const a of args) {
    const wrapped = wrapLine(`${argPad}${a}`);
    wrapped[wrapped.length - 1] += ",";
    out.push(...wrapped);
  }
  out.push(`${pad}${closer}${tail}`);
  return out;
}
