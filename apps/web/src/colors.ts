import type { Category } from '@netloom/blocks';

/** Okabe–Ito palette: distinguishable with the common forms of colour blindness. */
export const CATEGORY_COLORS: Record<Category, string> = {
  Inputs: '#0072B2',
  Layers: '#56B4E9',
  'CNN blocks': '#009E73',
  'Seq & attn': '#CC79A7',
  Backbones: '#E69F00',
  'Necks/merge': '#B8A600',
  Heads: '#D55E00',
  Detectors: '#882255',
  'My blocks': '#7A7A7A',
};
