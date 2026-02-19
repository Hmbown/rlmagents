# Release Process

This document describes the release process for the `rlmagents` package in this repository using [release-please](https://github.com/googleapis/release-please).

## Overview

Package releases are managed via release-please, which:

1. Analyzes conventional commits on the `main` branch
2. Creates/updates a release PR with changelog and version bump
3. When merged, creates a GitHub release
4. The release workflow publishes to PyPI and uploads artifacts

## How It Works

### Automatic Release PRs

When commits land on `main`, release-please analyzes them and either:

- **Creates a new release PR** if releasable changes exist
- **Updates an existing release PR** with additional changes
- **Does nothing** if no releasable commits are found (e.g. commits with type `chore`, `refactor`, etc.)

Release PRs are created on branches named `release-please--branches--main--components--<package>`.

### Triggering a Release

To trigger a package release:

1. Merge conventional commits to `main` (see [Commit Format](#commit-format))
2. Wait for release-please to create/update the release PR
3. Review the generated changelog in the PR
4. Merge the release PR — this creates a GitHub release
5. Review and edit the release notes in the GitHub UI
6. Publishing is handled through `.github/workflows/release.yml`, and PyPI upload is triggered from the same release flow.

### Version Bumping

Version bumps are determined by commit types:

| Commit Type                    | Version Bump  | Example                                  |
| ------------------------------ | ------------- | ---------------------------------------- |
| `fix:`                         | Patch (0.0.x) | `fix(rlmagents): resolve config loading issue` |
| `feat:`                        | Minor (0.x.0) | `feat(rlmagents): add new export command`      |
| `feat!:` or `BREAKING CHANGE:` | Major (x.0.0) | `feat(rlmagents)!: redesign config format`     |

> [!NOTE]
> While version is < 1.0.0, `bump-minor-pre-major` and `bump-patch-for-minor-pre-major` are enabled, so breaking changes bump minor and features bump patch.

## Commit Format

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/) format with types and scopes defined in `.github/workflows/pr_lint.yml`:

```text
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Examples

```bash
# Patch release
fix(rlmagents): resolve type hinting issue

# Minor release
feat(rlmagents): add new chat completion feature

# Major release (breaking change)
feat(rlmagents)!: redesign configuration format

BREAKING CHANGE: Config files now use TOML instead of JSON.
```

## Configuration Files

### `release-please-config.json`

Defines release-please behavior for each package.

### `.release-please-manifest.json`

Tracks the current version of each package:

```json
{
  "libs/rlmagents": "0.0.1"
}
```

This file is automatically updated by release-please when releases are created.

## Release Workflow

### Detection Mechanism

The release-please workflow (`.github/workflows/release-please.yml`) detects rlmagents releases by checking whether `libs/rlmagents/CHANGELOG.md` changed in the merge commit. This file is always updated by release-please when release PRs are merged.

### Lockfile Updates

When release-please creates or updates a release PR, the `update-lockfiles` job automatically regenerates `uv.lock` files since release-please updates `pyproject.toml` versions but doesn't regenerate lockfiles. Keep lockfiles in sync to avoid stale dependency resolution.

### Release Pipeline

The release workflow (`.github/workflows/release.yml`) runs when a release PR is merged:

1. **Build** - Creates distribution package
2. **Collect Contributors** - Gathers PR authors for release notes, including social media handles. Internal contributors are filtered separately.
3. **Release Notes** - Extracts changelog or generates from git log
4. **Test PyPI** - Publishes to test.pypi.org for validation
5. **Pre-release Checks** - Runs tests against the built package
6. **Mark Release** - Creates a GitHub release with the built artifacts
This is a published release (not draft) for `rlmagents`.

### Release PR Labels

Release-please uses labels to track the state of release PRs:

| Label | Meaning |
| ----- | ------- |
| `autorelease: pending` | Release PR has been merged but not yet tagged/released |
| `autorelease: tagged` | Release PR has been successfully tagged and released |

Because `skip-github-release: true` is set in the release-please config (we create releases via our own workflow instead of release-please), our `release.yml` workflow must update these labels manually. After successfully creating the GitHub release and tag, the `mark-release` job transitions the label from `pending` to `tagged`.

This label transition signals to release-please that the merged PR has been fully processed, allowing it to create new release PRs for subsequent commits.

## Manual Release

For hotfixes or exceptional cases, you can trigger a release manually. Use the `hotfix` commit type so as to not trigger a further PR update/version bump.

1. Go to **Actions** > **Package Release**
2. Click **Run workflow**
3. Select `rlmagents`
4. (Optionally enable `dangerous-nonmain-release` for hotfix branches)

> [!WARNING]
> Manual releases should be rare. Prefer the standard release-please flow.

## Troubleshooting

### "Found release tag with component X, but not configured in manifest" Warnings

You may see warnings in the release-please logs like:

```txt
⚠ Found release tag with an unmanaged component, but not configured in manifest
```

This is **harmless**. Release-please scans existing tags in the repository and warns when it finds tags for components that aren't in the current configuration. In this repository, tags from historical package attempts may surface as warnings until they are no longer present.

These warnings can be safely ignored as long as the tag is expected legacy noise.

### Unexpected Commit Authors in Release PRs

When viewing a release-please PR on GitHub, you may see commits attributed to contributors who didn't directly push to that PR. For example:

```txt
johndoe and others added 3 commits 4 minutes ago
```

This is a **GitHub UI quirk** caused by force pushes/rebasing, not actual commits to the PR branch.

**What's happening:**

1. release-please rebases its branch onto the latest `main`
2. The PR branch now includes commits from `main` as parent commits
3. GitHub's UI shows all "new" commits that appeared after the force push, including rebased parents

**The actual PR commits** are only:

- The release commit (e.g., `release(rlmagents): 0.0.18`)
- The lockfile update commit (e.g., `chore: update lockfiles`)

Other commits shown are just the base that the PR branch was rebased onto. This is normal behavior and doesn't indicate unauthorized access.

### Release PR Stuck with "autorelease: pending" Label

If a release PR shows `autorelease: pending` after the release workflow completed, the label update step may have failed. This can block release-please from creating new release PRs.

**To fix manually:**

```bash
# Find the PR number for the release commit
gh pr list --state merged --search "release(rlmagents)" --limit 5

# Update the label
gh pr edit <PR_NUMBER> --remove-label "autorelease: pending" --add-label "autorelease: tagged"
```

The label update is non-fatal in the workflow (`|| true`), so the release itself succeeded—only the label needs fixing.

### Yanking a Release

If you need to yank (retract) a release:

#### 1. Yank from PyPI

Using the PyPI web interface or a CLI tool.

#### 2. Delete GitHub Release/Tag (optional)

```bash
# Delete the GitHub release
gh release delete "rlmagents==<VERSION>" --yes

# Delete the git tag
git tag -d "rlmagents==<VERSION>"
git push origin --delete "rlmagents==<VERSION>"
```

#### 3. Fix the Manifest

Edit `.release-please-manifest.json` to the last good version:

```json
{
  "libs/rlmagents": "0.0.1"
}
```

Also update `libs/rlmagents/pyproject.toml` to match.

### Re-releasing a Version

PyPI does not allow re-uploading the same version. If a release failed partway:

1. If already on PyPI: bump the version and release again
2. If only on test PyPI: the workflow uses `skip-existing: true`, so re-running should work
3. If the GitHub release exists but PyPI publish failed: delete the release/tag and re-run the workflow

### "Untagged, merged release PRs outstanding" Error

If release-please logs show:

```txt
⚠ There are untagged, merged release PRs outstanding - aborting
```

This means a release PR was merged but its merge commit doesn't have the expected tag. This can happen if:

- The release workflow failed and the tag was manually created on a different commit (e.g., a hotfix)
- Someone manually moved or recreated a tag

**To diagnose**, compare the tag's commit with the release PR's merge commit:

```bash
# Find what commit the tag points to
git ls-remote --tags origin | grep "rlmagents==<VERSION>"

# Find the release PR's merge commit
gh pr view <PR_NUMBER> --json mergeCommit --jq '.mergeCommit.oid'
```

If these differ, release-please is confused.

**To fix**, move the tag and update the GitHub release:

```bash
# 1. Delete the remote tag
git push origin :refs/tags/rlmagents==<VERSION>

# 2. Delete local tag if it exists
git tag -d rlmagents==<VERSION> 2>/dev/null || true

# 3. Create tag on the correct commit (the release PR's merge commit)
git tag rlmagents==<VERSION> <MERGE_COMMIT_SHA>

# 4. Push the new tag
git push origin rlmagents==<VERSION>

# 5. Update the GitHub release's target_commitish to match
#    (moving a tag doesn't update this field automatically)
gh api -X PATCH repos/<owner>/<repo>/releases/$(gh api repos/<owner>/<repo>/releases --jq '.[] | select(.tag_name == "rlmagents==<VERSION>") | .id') \
  -f target_commitish=<MERGE_COMMIT_SHA>
```

After fixing, the next push to main should properly create new release PRs.

> [!NOTE]
> Moving a tag may require updating release metadata to match the correct commit. If the package was already published to PyPI, you can safely re-run publish — the publish workflow uses `skip-existing: true`, so it will succeed without re-uploading.

## References

- [release-please documentation](https://github.com/googleapis/release-please)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
