/// <reference types="vite/client" />

declare module 'monaco-editor/esm/vs/editor/editor.api' {
  export * from 'monaco-editor';
}
declare module 'monaco-editor/esm/vs/basic-languages/*';
declare module 'elkjs/lib/elk.bundled.js' {
  import ELK from 'elkjs';
  export default ELK;
}
