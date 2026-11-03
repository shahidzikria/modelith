import type { Display } from '@modelith/preprocess';
import { useEffect, useRef } from 'react';

/** Draws a Display (RGBA) into a canvas scaled to fit its box. */
export function Thumb({ display, className = '', label }: { display?: Display; className?: string; label?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !display) return;
    c.width = display.w;
    c.height = display.h;
    c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(display.rgba), display.w, display.h), 0, 0);
  }, [display]);
  return <canvas ref={ref} role="img" aria-label={label} className={`checker block max-h-full max-w-full object-contain ${className}`} />;
}
