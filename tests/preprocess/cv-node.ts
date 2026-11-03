// Load OpenCV.js (the same build the browser worker uses) in Node.
import { createRequire } from 'node:module';

export async function loadCv(): Promise<any> {
  const require = createRequire(import.meta.url);
  const mod = require('@techstark/opencv-js');
  if (mod.Mat) return mod;
  // The Emscripten module is a "thenable" that resolves to itself; awaiting it directly never settles.
  return new Promise((resolve) => {
    const done = (m: any) => {
      delete m.then;
      resolve(m);
    };
    if (typeof mod.then === 'function') mod.then(done);
    else mod.onRuntimeInitialized = () => done(mod);
  });
}
