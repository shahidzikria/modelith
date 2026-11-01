import { analyze, type Analysis, type Project } from '@netloom/blocks';
import { useEffect, useState } from 'react';
import { useStore } from './store';

/** Shape inference re-run on every edit, debounced by 50 ms. */
export function useAnalysis(): Analysis {
  const project = useStore((s) => s.project);
  const [result, setResult] = useState<Analysis>(() => run(project));
  useEffect(() => {
    const t = setTimeout(() => setResult(run(project)), 50);
    return () => clearTimeout(t);
  }, [project]);
  return result;
}

function run(p: Project): Analysis {
  return analyze({ nodes: p.nodes, edges: p.edges, framework: p.framework });
}
