# Manual releases for `@propio-ai/providers`

`main` is the only release branch. Every push to `main` runs the **Main build**
workflow and uploads an inspectable package candidate. Publication is a separate
maintainer action: the **Release** workflow runs only when manually dispatched,
rebuilds the selected current `main` commit, and uses npm Trusted Publishing.

No release commit, changelog commit, npm token, or local `npm publish` command is
used.

## Commit and merge policy

Pull requests targeting `main` must pass the `required` CI job. The repository
must allow only squash merges, configured to use the PR title as the squash
commit title (`PR_TITLE`) and the PR body as its body (`PR_BODY`). CI validates
those exact fields, so the squash commit is the release unit.

Use Conventional Commit titles, for example:

```text
fix: preserve a provider response field
feat(openai): support a new request option
docs: clarify configuration
```

`fix:` triggers a patch release and `feat:` triggers a minor release. Other
valid conventional types, such as `docs:`, `test:`, `chore:`, and `ci:`, do not
release by default.

This package remains pre-1.0. Until a deliberate stabilization decision,
breaking markers are rejected in CI: do not use `!` before the colon in a PR
title and do not add a `BREAKING CHANGE:` footer to a PR body.

Several merges may be included in one manually authorized release. Semantic
Release evaluates every commit since the latest `v<version>` tag and chooses the
highest required version bump.

## Main-build artifact

After every push to `main`, **Main build** repeats the Node.js 24.10.0
validation, runs the whole-repository Fallow check, builds a tarball, records its
SHA-256 checksum and npm pack metadata, and uploads
`npm-package-candidate-<commit>` for 30 days.

The candidate is for inspection only. Its metadata explicitly sets
`publishable: false` because its package version is still the source version in
`main`. Do not publish that tarball. The manual release rebuilds after Semantic
Release computes the final version.

## Initial rollout

1. Verify that npm's `@propio-ai/providers@0.1.4` `gitHead` is
   `66d4cdf53e3efcb0bd391afd83944a3a7060ece5`, then create and push `v0.1.4`
   at that exact published commit. Later non-release commits must remain after
   the baseline tag.
2. Open and validate the workflow pull request. Observe that the aggregate job
   context is literally `required` (the GitHub UI may display `CI / required`).
3. Configure `main` to require pull requests and the `required` status check,
   with zero approvals while this is a solo-maintainer repository. Disable
   merge commits and rebase merges; retain squash merges with `PR_TITLE` and
   `PR_BODY`.
4. Add a `v*` tag ruleset that prevents updating or deleting existing tags but
   permits creation of new release tags. Do not grant the GitHub Actions app a
   repository-wide ruleset bypass.
5. In npm package settings, configure a Trusted Publisher for package
   `@propio-ai/providers`, owner/repository `esack7/propio-providers`, and
   workflow file `release.yml`.
6. Set the repository Actions variable `NPM_PUBLISH_ENABLED` to `true`. This is
   an emergency kill switch; setting it does not publish anything.
7. From **Actions → Release → Run workflow**, select `main`, choose
   `mode=dry-run`, and optionally provide the current full commit SHA. Confirm
   the expected version and `OIDC token exchange with the npm registry
   succeeded`. No package, tag, or GitHub Release is created.
8. Inspect the matching Main-build artifact. When ready, dispatch **Release**
   from `main` with `mode=publish` and the current 40-character `main` SHA in
   `expected_sha`. Verify npm provenance, package contents, the `v<version>`
   tag, and the GitHub Release.

## Manual release controls

The workflow refuses live publication unless all of these conditions hold:

- it was manually dispatched from `main`;
- the selected commit is the current remote `main` both before and after
  validation;
- `mode` is `publish`;
- `expected_sha` exactly matches that current commit; and
- `NPM_PUBLISH_ENABLED` is exactly `true`.

The release workflow repeats formatting, build, unit tests, and `npm pack
--dry-run` on Node.js 24.10.0 with npm 11.5.1 before Semantic Release can
publish. Repository-local concurrency prevents overlapping manual releases.

Credentialed `npm run test:integration` remains a manual developer check and is
not part of required CI or the release workflow.

## Dry runs, disabling, and recovery

Use `mode=dry-run` to preview the next version and release notes. A dry run
neither publishes nor creates tags or a GitHub Release. It still verifies npm
OIDC and repository push access.

To disable live releases, set `NPM_PUBLISH_ENABLED` to anything other than
`true`; do not delete the workflow or release history. If a dry run fails with
`ENONPMTOKEN`, do not add a token. Inspect earlier log lines for `OIDC token
exchange with the npm registry failed` and check the npm Trusted Publisher
owner, repository, workflow filename, GitHub-hosted runner, and
`id-token: write` permission.

If publication fails after a tag or package version may have been created,
inspect npm, GitHub Releases, and tags before rerunning. Do not manually reuse
or republish an immutable npm version. Semantic Release tags and npm metadata,
not the version in `main`, are the release source of truth.
