# Fallow baseline

This repository pins `fallow` at `3.9.1` in `devDependencies`. Run `npm ci` before analysis so Fallow resolves the repository's dependency graph.

```bash
npm run fallow:check
```

The script executes the complete repository command `fallow --fail-on-issues` and must finish with zero issues. The retry and HTTP-status translation paths are decomposed into focused helpers so the repository meets Fallow's default health thresholds without local exceptions.

For a strict pull-request delta gate, run:

```bash
npm run fallow:audit
```

That script executes `fallow audit --gate all`. It is intentionally available for a subsequent CI change; this cleanup does not alter release CI.
