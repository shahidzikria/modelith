// Line-based source builder that remembers which pipeline step produced each line (code ↔ pipeline linking).
import type { GeneratedFile } from '../types';

export class Source {
  private lines: string[] = [];
  private owners: (string | undefined)[] = [];

  /** Append text (may contain newlines) at an indent level, tagging every line with a step uid. */
  add(text: string, indentLevel = 0, uid?: string): this {
    const pad = '    '.repeat(indentLevel);
    for (const raw of text.split('\n')) {
      this.lines.push(raw.trim() ? pad + raw : '');
      this.owners.push(uid);
    }
    return this;
  }

  blank(n = 1): this {
    for (let i = 0; i < n; i++) {
      this.lines.push('');
      this.owners.push(undefined);
    }
    return this;
  }

  file(path: string, language: GeneratedFile['language'] = 'python'): GeneratedFile {
    const outLines: string[] = [];
    const lineMap: Record<number, string> = {};
    let blanks = 0;
    this.lines.forEach((l, i) => {
      if (!l.trim()) {
        blanks++;
        if (blanks > 2) return;
      } else blanks = 0;
      outLines.push(l.replace(/\s+$/, ''));
      const owner = this.owners[i];
      if (owner) lineMap[outLines.length] = owner;
    });
    while (outLines.length && !outLines[outLines.length - 1].trim()) outLines.pop();
    return { path, content: outLines.join('\n') + '\n', lineMap, language };
  }
}
