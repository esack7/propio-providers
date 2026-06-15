# Publishing `@propio-ai/providers`

This document describes the repeatable workflow for publishing the npm package.

## Package Identity

- npm package name: `@propio-ai/providers`
- Package format: ESM only
- Published entrypoint: `dist/index.js`
- Published types: `dist/index.d.ts`
- Minimum runtime: Node.js >= 20

## When To Publish

Publish when you:

- Add or change provider behavior
- Add, remove, or change supported public API exports
- Change dependencies or dependency versions
- Update public README documentation
- Bump the release version

## Version Bump

Use one of the standard npm version commands, but make the version bump on a
release branch instead of committing directly to `main`:

```bash
git checkout main
git pull --ff-only
git checkout -b release/v<version>
npm version patch --no-git-tag-version
```

Use `minor` or `major` instead of `patch` when the change warrants it. Replace
`<version>` in the branch name with the version you are releasing, for example
`release/v0.1.1`.

The `--no-git-tag-version` flag keeps npm from creating a local release commit
and tag before the protected-branch PR has merged. The version bump should update
both:

- `package.json`
- `package-lock.json`

Commit those files, push the release branch, and merge it through the normal PR
flow:

```bash
git status --short
git add package.json package-lock.json
git commit -m "chore: release v<version>"
git push -u origin release/v<version>
```

Do not publish from the release branch. Publish only after the version bump PR is
merged and your local `main` contains the release commit:

```bash
git checkout main
git pull --ff-only
```

If this repository uses git tags for releases, create and push the tag after the
PR has merged:

```bash
git tag v<version>
git push origin v<version>
```

## Dependency Updates

If you add, remove, or update dependencies:

```bash
npm install <package>@<version>
```

Commit the resulting `package.json` and `package-lock.json` changes. This repo
does not currently publish `npm-shrinkwrap.json`; npm uses `package-lock.json`
for local reproducibility, while consumers resolve dependencies from the
published dependency ranges.

## Pre-Publish Checklist

Run the release checks from the repository root:

```bash
npm test
npm run build
npm run format:check
npm pack --dry-run
```

Run the integration suite only when live provider behavior needs validation and
the required credentials are available:

```bash
npm run test:integration
```

The `npm pack --dry-run` output should include:

- `dist/`
- `dist/index.js`
- `dist/index.d.ts`
- `README.md`
- `LICENSE`
- `package.json`

The tarball should not include source files, tests, local environment files, or
generated working artifacts outside `dist/`.

## Smoke Test

Create a tarball and install it in a clean temp directory:

```bash
npm pack
mkdir -p /tmp/propio-providers-release-test
cd /tmp/propio-providers-release-test
npm init -y
npm pkg set type=module
npm install /path/to/propio-ai-providers-<version>.tgz
```

Then verify the package can be imported as ESM:

```bash
node --input-type=module -e "import('@propio-ai/providers').then((m) => { console.log(typeof m.createProvider); console.log(typeof m.validateProvidersConfig); })"
```

The key checks are:

- The import exits cleanly under Node.js >= 20
- `createProvider` is exported as a function
- `validateProvidersConfig` is exported as a function

If the release changes provider behavior, also run a targeted consumer-style
check for the affected provider using environment variables for credentials.
Avoid hardcoding secrets in the test directory or shell history.

## Publish

When the version bump PR has merged into `main` and the tarball and smoke test
look good:

```bash
npm publish --access public
```

The package already has `publishConfig.access` set to `public`, but passing the
flag keeps the publish command explicit.

## After Publish

- Confirm the package page and version on npm
- Confirm a clean install works with `npm install @propio-ai/providers@<version>`
- Record the published version in the release notes or changelog if the repo uses one
- If the next release changes dependencies, refresh `package-lock.json` again before publishing
