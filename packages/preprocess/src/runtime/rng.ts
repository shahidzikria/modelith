// NumPy-compatible random numbers: SeedSequence + PCG64 + Generator.random(), so a seeded augmentation
// previews exactly what `np.random.Generator(np.random.PCG64(seed))` produces in the exported code.
// Generated Python only ever draws from `rng.random()`; uniform / integer / normal helpers are derived from
// it identically on both sides (see codegen/helpers.ts).

const M32 = 0xffffffff;
const M64 = (1n << 64n) - 1n;
const M128 = (1n << 128n) - 1n;
const PCG_MULT = (0x2360ed051fc65da4n << 64n) | 0x4385df649fccf645n;

const INIT_A = 0x43b0d7e5;
const MULT_A = 0x931e8875;
const INIT_B = 0x8b51f9dd;
const MULT_B = 0x58f38ded;
const MIX_MULT_L = 0xca01f9dd;
const MIX_MULT_R = 0x4973f715;
const XSHIFT = 16;
const POOL_SIZE = 4;

const mul32 = (a: number, b: number) => Math.imul(a, b) >>> 0;

/** numpy.random.SeedSequence(seed).generate_state(nWords, uint64) for a non-negative integer seed. */
export function seedSequenceState(seed: number, nWords64: number): bigint[] {
  // Entropy as little-endian uint32 words.
  let s = BigInt(Math.floor(Math.abs(seed)));
  const entropy: number[] = [];
  do {
    entropy.push(Number(s & 0xffffffffn));
    s >>= 32n;
  } while (s > 0n);

  const hashConst = [INIT_A];
  const hashmix = (value: number): number => {
    value = (value ^ hashConst[0]) >>> 0;
    hashConst[0] = mul32(hashConst[0], MULT_A);
    value = mul32(value, hashConst[0]);
    return (value ^ (value >>> XSHIFT)) >>> 0;
  };
  const mix = (x: number, y: number): number => {
    let r = (mul32(MIX_MULT_L, x) - mul32(MIX_MULT_R, y)) >>> 0;
    r = (r ^ (r >>> XSHIFT)) >>> 0;
    return r;
  };
  const pool = new Array<number>(POOL_SIZE).fill(0);
  for (let i = 0; i < POOL_SIZE; i++) pool[i] = hashmix(i < entropy.length ? entropy[i] : 0);
  for (let src = 0; src < POOL_SIZE; src++)
    for (let dst = 0; dst < POOL_SIZE; dst++) if (src !== dst) pool[dst] = mix(pool[dst], hashmix(pool[src]));
  for (let src = POOL_SIZE; src < entropy.length; src++)
    for (let dst = 0; dst < POOL_SIZE; dst++) pool[dst] = mix(pool[dst], hashmix(entropy[src]));

  let hc = INIT_B;
  const words: number[] = [];
  for (let i = 0; i < nWords64 * 2; i++) {
    let v = (pool[i % POOL_SIZE] ^ hc) >>> 0;
    hc = mul32(hc, MULT_B);
    v = mul32(v, hc);
    v = (v ^ (v >>> XSHIFT)) >>> 0;
    words.push(v);
  }
  const out: bigint[] = [];
  for (let i = 0; i < nWords64; i++) out.push((BigInt(words[2 * i + 1]) << 32n) | BigInt(words[2 * i]));
  return out;
}

export interface RngState {
  state: string;
  inc: string;
}

/** PCG64 (XSL-RR 128/64) seeded like numpy.random.PCG64(seed). */
export class Rng {
  private state = 0n;
  private inc = 0n;

  constructor(seed: number | RngState) {
    if (typeof seed === 'number') {
      const v = seedSequenceState(seed, 4);
      const initState = (v[0] << 64n) | v[1];
      const initSeq = (v[2] << 64n) | v[3];
      this.inc = ((initSeq << 1n) | 1n) & M128;
      this.step();
      this.state = (this.state + initState) & M128;
      this.step();
    } else {
      this.state = BigInt(seed.state);
      this.inc = BigInt(seed.inc);
    }
  }

  private step(): void {
    this.state = (this.state * PCG_MULT + this.inc) & M128;
  }

  nextUint64(): bigint {
    this.step();
    const hi = this.state >> 64n;
    const x = (hi ^ this.state) & M64;
    const rot = hi >> 58n;
    return ((x >> rot) | (x << ((64n - rot) & 63n))) & M64;
  }

  /** Generator.random(): float64 in [0, 1). */
  random(): number {
    return Number(this.nextUint64() >> 11n) * 2 ** -53;
  }

  /** Generator.random(n), C order. */
  randomArray(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.random();
    return out;
  }

  /** `_uniform(rng, lo, hi)` in the generated helpers. */
  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.random();
  }

  /** `_randint(rng, lo, hi)`: integer in [lo, hi] inclusive. */
  randint(lo: number, hi: number): number {
    return lo + Math.min(Math.floor(this.random() * (hi - lo + 1)), hi - lo);
  }

  /** `_chance(rng, p)`; always draws so the stream does not depend on p. */
  chance(p: number): boolean {
    return this.random() < p;
  }

  /** State as Python's `rng.bit_generator.state` (decimal strings), for continuing on the server. */
  getState(): RngState {
    return { state: this.state.toString(), inc: this.inc.toString() };
  }

  clone(): Rng {
    return new Rng(this.getState());
  }
}

export { M32 };
