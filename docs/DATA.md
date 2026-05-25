# Data

## Source

The analysis uses the **titanic5** dataset, curated by Encyclopedia Titanica and hosted by Vanderbilt Biostatistics.

- **Direct URL:** [https://hbiostat.org/data/repo/titanic5.csv](https://hbiostat.org/data/repo/titanic5.csv)
- **Curator:** [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/)
- **Host:** [hbiostat.org](https://hbiostat.org/data/)
- **Size:** 1,309 rows × 14 columns (~95 KB CSV)
- **Local path:** `data/raw/titanic5.csv` (gitignored)

Fetch it with `make download` or `python scripts/download_data.py`.

---

## Column dictionary

### Raw columns (14)

| Column | Type | Description | Missing |
|---|---|---|---|
| `PassengerId` | int | Unique row identifier (1-based) | 0 |
| `Survived` | int (0/1) | Target outcome — **1 = survived, 0 = perished** | 0 |
| `Pclass` | int (1/2/3) | Ticket class — **1 = upper, 2 = middle, 3 = lower** | 0 |
| `Name` | str | "Last, Title Firstname" — Title is extractable | 0 |
| `Sex` | str | `female` / `male` | 0 |
| `Age` | float | Years (fractional for infants) | **51 (3.9%)** |
| `SibSp` | int | Siblings + spouse aboard | 0 |
| `Parch` | int | Parents + children aboard | 0 |
| `Ticket` | str | Ticket number (mixed format) | 0 |
| `Fare` | float | Passenger fare in USD (1912 dollars) | 0 |
| `Embarked` | str | `C` Cherbourg, `Q` Queenstown, `S` Southampton, `B` Belfast (crew) | 2 |
| `Occupation` | str | Recorded occupation; sparse | **621 (47.4%)** |
| `BoatBody` | str | Lifeboat number (e.g. `"5"`, `"A"`) **or** body recovery code in brackets (e.g. `"[190]"`) — blank string if neither | 0 (but many blanks) |
| `NameId` | int | Encyclopedia Titanica internal ID | 0 |

### Engineered features (added by `engineer_features`)

| Column | Type | How computed |
|---|---|---|
| `Title` | str | Regex on `Name`: `Mr`, `Mrs`, `Miss`, `Master`, `Officer`, `Royalty`, `Other` |
| `FamilySize` | int | `SibSp + Parch + 1` |
| `IsAlone` | int (0/1) | `1` if `FamilySize == 1` |
| `AgeGroup` | category | `Child` (≤16), `Teen` (17–25), `Adult` (26–50), `Senior` (51+) |
| `FareGroup` | category | Quartile bins: `Low`, `Medium`, `High`, `Very High` |
| `FarePerPerson` | float | `Fare / FamilySize` |
| `HasCabin` | int (0/1) | `1` if `Occupation` is non-null (low-resolution cabin-info proxy) |
| `Lifeboat` | str / None | Parsed from `BoatBody` — set only when `BoatBody` is non-blank and not bracketed |
| `BodyRecovered` | int (0/1) | `1` if `BoatBody` contained a bracketed body code |
| `Age` (imputed) | float | Median Age within Sex × Pclass strata fills the 51 missing values |

---

## Cleaning rules

`clean_data(df)` — minimal, non-destructive:

1. `Embarked` — 2 missing rows filled with the **mode** (`S` Southampton).
2. `Fare` — any missing values filled with the **median fare within each `Pclass`**.
3. `Age` — **deliberately left missing** here; it gets stratum-aware imputation downstream in `engineer_features`.

`engineer_features(df, impute_age=True)` — adds derived features and imputes Age. The `impute_age=False` switch is available for analyses that need to respect missingness explicitly.

---

## Quirks worth knowing

- **Pre-decimal British fare format** (e.g., `"£211 60s 9d"`) appears in some raw values. The loader's `_parse_price()` converts them to decimal USD.
- **Belfast (`B`) embarked** — these are 10 crew/staff who joined at the shipyard before the maiden voyage. Their survival rate is 0%.
- **BoatBody is a polymorphic column.** Same column encodes two very different events. The engineered split (`Lifeboat` vs `BodyRecovered`) makes that explicit.
- **Lifeboat is downstream of survival.** Almost all passengers with a lifeboat number survived (98.6%) — using `Lifeboat` as a feature in a predictive model would be target leakage. See [METHODOLOGY.md](METHODOLOGY.md).

---

## Counts that should reconcile

Anywhere this dataset is summarised, totals should add to **1,309**:

- Sex: female 466 + male 843 = **1,309**
- Class: 1st 324 + 2nd 276 + 3rd 709 = **1,309**
- Embarked (after imputing the 2 NAs to `S`): C 272 + Q 123 + S 904 + B 10 = **1,309**
- Joint Class × Sex: (1F 144) + (1M 180) + (2F 106) + (2M 170) + (3F 216) + (3M 493) = **1,309**
- Family size 1–11 totals = **1,309**

If you ever see different subtotals on the same surface, that's a sign you're mixing raw and engineered DataFrames. Run `python reports/validate_dataset.py` to confirm.
