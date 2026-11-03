/** Switch between the model builder and the Preprocessing Studio (hash routes). */
export function ViewSwitch({ active }: { active: 'model' | 'studio' }) {
  return (
    <nav className="flex rounded-md border border-line p-0.5 text-xs" aria-label="Modelith modules">
      <a href="#/" aria-current={active === 'model' ? 'page' : undefined} className={`rounded px-2 py-1 ${active === 'model' ? 'bg-accent text-accent-fg' : 'text-muted hover:text-fg'}`}>
        Model
      </a>
      <a href="#/preprocess" aria-current={active === 'studio' ? 'page' : undefined} className={`rounded px-2 py-1 whitespace-nowrap ${active === 'studio' ? 'bg-accent text-accent-fg' : 'text-muted hover:text-fg'}`}>
        <span className="sm:hidden">Prep</span>
        <span className="hidden sm:inline">Preprocessing</span>
      </a>
    </nav>
  );
}
