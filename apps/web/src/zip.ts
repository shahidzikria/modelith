import { ident, type GeneratedFile } from '@modelith/blocks';
import JSZip from 'jszip';

export function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function downloadZip(projectName: string, files: GeneratedFile[]): Promise<void> {
  const zip = new JSZip();
  const folder = zip.folder(ident(projectName))!;
  for (const f of files) folder.file(f.path, f.content);
  const blob = await zip.generateAsync({ type: 'blob' });
  download(blob, `${ident(projectName)}.zip`);
}
