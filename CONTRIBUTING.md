# Contributing

Thanks for considering a contribution. This is a personal portfolio project, so the bar for changes is "does this make the analysis more correct, more readable, or more reproducible?"

---

## Quick checks before opening a PR

```bash
make test          # all 49 tests must pass
make validate      # row counts reconcile to 1,309 at every stage
ruff check src tests scripts dashboard reports
ruff format --check src tests scripts dashboard reports
```

The pre-commit hook runs ruff + pytest automatically; see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#pre-commit-hooks).

---

## What's in scope

- **Bug fixes** in the data pipeline, statistics, or output generators.
- **Documentation** improvements — typos, clearer wording, missing details.
- **New analyses** that fit the project's narrative (effect sizes, odds ratios, statistical tests, visualisations).
- **Better engineering** — refactors, type hints, more tests, cleaner abstractions.

## What's likely out of scope

- **A full ML pipeline.** That's tracked in [ROADMAP.md](ROADMAP.md) but kept separate to keep this repo focused on classical inference + EDA. A predictive-modelling sibling repo would be the cleaner home for that work.
- **Cosmetic style debates** about the dashboard or report layout — I've tuned them to my taste, but I'm open to objectively-better changes (accessibility, contrast, mobile rendering).

---

## Style & conventions

- **Line length 100.** Enforced by ruff.
- **Type hints** on public functions in `src/`.
- **Docstrings:** one-line summary minimum on public functions. Longer when behaviour is non-obvious or when reproducing a quirk of the dataset.
- **Tests live in `tests/test_<module>.py`** mirroring the `src/` layout.
- **No comments that paraphrase code.** Comments explain *why*, not *what*.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for conventions and common tasks.

---

## Commit messages

```
<verb> <what changed> (<why if not obvious>)
```

Examples that work:

- `Add Wilson CI helpers to src/analysis/inference.py`
- `Fix _parse_boat_body so whitespace strings don't become lifeboat numbers`
- `Tune light-theme palette for chart-label legibility`

Examples that don't:

- `update stuff`
- `wip`

---

## Reporting issues

Open an issue with:

1. What you expected to happen
2. What actually happened
3. The minimum command sequence to reproduce
4. Python + OS versions

For data-correctness bugs especially, include the output of `python reports/validate_dataset.py` — it's the canonical sanity check.
