// Minimal .npy reader / writer (C order, little endian) for exchanging arrays with pytest.
import { readFileSync, writeFileSync } from 'node:fs';

type Arr = Uint8Array | Uint16Array | Float32Array | Float64Array | Int32Array;

const DESCR: [string, new (b: ArrayBuffer) => Arr][] = [
  ['<u1', Uint8Array as any],
  ['|u1', Uint8Array as any],
  ['<u2', Uint16Array as any],
  ['<f4', Float32Array as any],
  ['<f8', Float64Array as any],
  ['<i4', Int32Array as any],
];

function descrOf(a: Arr): string {
  if (a instanceof Uint8Array) return '|u1';
  if (a instanceof Uint16Array) return '<u2';
  if (a instanceof Float32Array) return '<f4';
  if (a instanceof Float64Array) return '<f8';
  return '<i4';
}

export function writeNpy(path: string, data: Arr, shape: number[]): void {
  let header = `{'descr': '${descrOf(data)}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const total = 10 + header.length + 1;
  header += ' '.repeat((64 - (total % 64)) % 64) + '\n';
  const head = Buffer.alloc(10);
  head.write('\x93NUMPY', 0, 'latin1');
  head[6] = 1;
  head[7] = 0;
  head.writeUInt16LE(header.length, 8);
  writeFileSync(path, Buffer.concat([head, Buffer.from(header, 'latin1'), Buffer.from(data.buffer, data.byteOffset, data.byteLength)]));
}

export function readNpy(path: string): { data: Arr; shape: number[] } {
  const b = readFileSync(path);
  const major = b[6];
  const hlen = major === 1 ? b.readUInt16LE(8) : b.readUInt32LE(8);
  const off = major === 1 ? 10 : 12;
  const header = b.subarray(off, off + hlen).toString('latin1');
  const descr = /'descr':\s*'([^']+)'/.exec(header)![1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)![1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const ctor = DESCR.find(([d]) => d === descr)?.[1];
  if (!ctor) throw new Error(`unsupported dtype ${descr}`);
  const raw = b.subarray(off + hlen);
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { data: new ctor(ab as ArrayBuffer), shape };
}
