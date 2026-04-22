# Delhivery feature engineering
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/delhivery-logistics-eda/blob/main/Delhivery_Logistics_CaseStudy.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-4C72B0.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6.svg)](https://scipy.org/)

End-to-end **logistics data engineering and EDA** pipeline for Delhivery's trip-level operational dataset.
Transforms raw segment-level delivery records into clean, aggregated trip features — comparing OSRM planned routes against actual behaviour, testing statistical hypotheses on delays, and surfacing business insights about corridor performance and route efficiency.

## Highlights

| Feature | Detail |
| --- | --- |
| **Two-level aggregation** | Segment-level → Trip-level via configurable `groupby` + `agg` pipeline |
| **Feature engineering** | Location parsing (city, state, place code), temporal features, OD duration, composite scores |
| **OSRM vs Actual analysis** | Wilcoxon signed-rank tests across time and distance fields; systematic underestimation quantified |
| **Outlier treatment** | IQR detection across 9 key numeric columns + Winsorization (clip to bounds) |
| **Categorical encoding** | One-hot encoding with `drop_first=True` for route type, source/dest states |
| **Feature scaling** | Both `StandardScaler` (z-score) and `MinMaxScaler` ([0,1]) applied and compared |
| **Hypothesis testing** | Wilcoxon tests — OD vs Scan time, Actual vs OSRM time, aggregation consistency checks |
| **Business insights** | Top 10 corridor heatmap, route-type performance breakdown, timing effect analysis |

## Architecture

```
                    ┌───────────────────────────────────────────── ┐
                    │             Raw Data Layer                   │
                    │   delhivery_data.csv  (46,413 rows × 24 cols)│
                    │   Segment-level trip records                 │
                    └─────────────────┬─────────────────────────── ┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │          Data Cleaning & Type Conversion    │
                    │  Datetime parsing · Category dtypes         │
                    │  Median imputation · 'Unknown' fill         │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │           Feature Engineering               │
                    │  Location parsing (city / state / code)     │
                    │  Temporal features (hour, weekday, month)   │
                    │  OD duration · Composite segment key        │
                    └──────────┬──────────────────┬──────────────┘
                               │                  │
          ┌────────────────────▼────┐   ┌─────────▼────────────────────┐
          │  Segment Aggregation    │   │   Trip Aggregation           │
          │  46,413 rows → 8,492    │   │   8,492 segments → 4,752     │
          │  SUM / FIRST / MIN-MAX  │   │   SUM / FIRST / LAST         │
          └────────────────────┬────┘   └─────────┬────────────────────┘
                               └────────┬─────────┘
                                        │
                    ┌───────────────────▼─────────────────────────┐
                    │         Hypothesis Testing                  │
                    │  Wilcoxon: OD vs Scan · Actual vs OSRM      │
                    │  Aggregation consistency validation         │
                    └───────────────────┬─────────────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
    ┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌─────────▼──────────────┐
    │  Outlier Treatment  │  │  Encoding & Scaling │  │  Business Insights     │
    │  IQR detection      │  │  One-hot + OHE      │  │  Corridor heatmap      │
    │  Winsorization      │  │  StandardScaler     │  │  Route type perf.      │
    │  9 numeric columns  │  │  MinMaxScaler       │  │  Timing effects        │
    └─────────────────────┘  └─────────────────────┘  └────────────────────────┘
```

---

## Project Structure

```
delhivery-logistics-eda/
│
├── Delhivery_Logistics_CaseStudy.ipynb   ← Main notebook (open in Colab)
├── delhivery_data.csv                    ← Raw dataset (46,413 rows × 24 cols)
└── README.md                             ← This file
```
## Dataset

| Column | Description | Type |
| --- | --- | --- |
| `data` | Train / test split indicator | category |
| `trip_creation_time` | Timestamp of trip creation | datetime |
| `route_schedule_uuid` | Route schedule identifier | object |
| `route_type` | FTL (Full Truck Load) or Carting | category |
| `trip_uuid` | Unique trip identifier | object |
| `source_center` / `source_name` | Origin hub code and name | category |
| `destination_center` / `destination_name` | Destination hub code and name | category |
| `od_start_time` / `od_end_time` | Gate-to-gate start and end timestamps | datetime |
| `start_scan_to_end_scan` | Duration from first to last scan (minutes) | float64 |
| `actual_distance_to_destination` | Actual distance travelled (km) | float64 |
| `actual_time` | Actual trip time (minutes) | float64 |
| `osrm_time` / `osrm_distance` | OSRM-planned time and distance | float64 |
| `factor` | Ratio of actual to OSRM time | float64 |
| `segment_actual_time` | Actual time for the segment | float64 |
| `segment_osrm_time` / `segment_osrm_distance` | OSRM predictions at segment level | float64 |

- **46,413 rows** (segment-level) → **4,752 trips** after two-stage aggregation
- Missing values only in `source_name` (121) and `destination_name` (76) — treated with `'Unknown'`

---

## Notebook Walkthrough

| # | Section | What It Does |
| --- | --- | --- |
| 1 | Import Libraries | pandas, numpy, seaborn, scipy, sklearn — consistent plot styling |
| 2 | Data Loading & Initial Exploration | Shape, dtypes, missing values, statistical summary |
| 3 | Data Type Conversions | Timestamps → datetime, identifiers → category dtype |
| 4 | Missing Value Treatment | Median imputation (numeric), 'Unknown' fill (categorical) |
| 5 | Feature Creation from Names | Parse `City-Place-Code (State)` pattern → `src_city`, `src_state`, `dest_city`, `dest_state` |
| 6 | Time-based Features | `trip_year`, `trip_month`, `trip_day`, `trip_hour`, `trip_weekday` from `trip_creation_time` |
| 7 | Segment-level Aggregation | `groupby(segment_key)` → 46,413 rows to 8,492 segments |
| 8 | Trip-level Aggregation | `groupby(trip_uuid)` → 8,492 segments to 4,752 trips |
| 9 | OD Duration Feature | `od_end_time − od_start_time` in minutes → `od_total_time` |
| 10 | Visual Analysis: OD vs Scan Time | Scatter + difference distribution + Wilcoxon test |
| 11 | Hypothesis Testing: Actual vs OSRM | Scatter vs perfect-prediction line + Wilcoxon test |
| 12 | Multiple Hypothesis Tests | Consistency checks: `actual_time` vs `segment_actual_time`, distance and time pairs |
| 13 | Outlier Detection & Treatment | IQR method → summary table → Winsorization (clip to bounds) |
| 14 | Categorical Encoding | `pd.get_dummies` with `drop_first=True` on route type and states |
| 15 | Feature Scaling | `StandardScaler` + `MinMaxScaler` — side-by-side comparison |
| 16 | Business Insights | Top 10 state corridors · Performance by route type · Timing effects |
| 17 | Actionable Recommendations | ETA recalibration · Corridor prioritisation · Route optimisation · Dispatch timing |

---

## Hypothesis Testing Summary

| Test | Comparison | p-value | Conclusion |
| --- | --- | --- | --- |
| Wilcoxon signed-rank | OD Total Time vs Scan-to-Scan Time | < 0.001 | Actual gate-to-gate times **exceed** scan times — operational delays exist between physical movement and scan events |
| Wilcoxon signed-rank | Actual Time vs OSRM Time | < 0.001 | OSRM **systematically underestimates** actual delivery times across all trips |
| Wilcoxon signed-rank | `actual_time` vs `segment_actual_time` | < 0.001 | Significant difference confirms aggregation captures real variation |
| Wilcoxon signed-rank | `osrm_distance` vs `segment_osrm_distance` | < 0.001 | Distance aggregation consistent — SUM logic validated |
| Wilcoxon signed-rank | `osrm_time` vs `segment_osrm_time` | < 0.001 | Time aggregation consistent — SUM logic validated |

---

## Key Business Insights

| # | Insight | Impact |
| --- | --- | --- |
| 1 | **Corridor concentration** — 70%+ of all trips occur on top 10 state pairs | Focus capacity planning and staffing on Maharashtra↔Delhi, Karnataka↔Tamil Nadu corridors |
| 2 | **OSRM underestimates by 20–40%** across all route types | Customer ETAs are systematically optimistic — add a correction factor to all delivery promises |
| 3 | **FTL shows lower time variability** than Carting | Consolidate stable-volume routes to FTL wherever feasible for reliability |
| 4 | **Peak-hour dispatches (18:00–22:00) inflate times by 25%+** | Schedule bulk long-haul dispatches 06:00–10:00 for best on-time performance |
| 5 | **5–15% of trips are outliers** (heavy right-tail) | Long-haul / delayed trips need a separate operational playbook and dedicated monitoring alerts |

---

## Results — Data Reduction Pipeline

| Stage | Rows | Description |
| --- | --- | --- |
| Raw dataset | 46,413 | Segment-level rows, one row per delivery scan event |
| After segment aggregation | 8,492 | One row per unique trip-source-destination segment |
| After trip aggregation | 4,752 | One row per complete end-to-end trip |
| After outlier treatment | 4,752 | All rows retained — extreme values capped, no deletions |
| After encoding + scaling | 4,752 | Model-ready feature matrix with 12 columns |

---

## Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python 3.10 |
| Data manipulation | pandas · NumPy |
| Visualisation | Matplotlib · Seaborn |
| Statistical testing | SciPy (Wilcoxon, IQR) |
| Preprocessing | scikit-learn (StandardScaler, MinMaxScaler) |
| Notebook environment | Google Colab|

## Actionable Recommendations

**1. ETA Recalibration**
Update all customer delivery promises using historical actual times. Add a 25–40% correction buffer on top of current OSRM-based ETAs until a recalibrated model is deployed.

**2. Corridor Prioritisation**
Double capacity and dedicated staffing on the top 10 state corridors. Create fixed line-haul schedules for high-volume routes to reduce variability.

**3. Route Optimisation**
Systematically migrate stable-volume routes from Carting → FTL. Monitor FTL truck utilisation and expand only when load factor consistently exceeds 80%.

**4. Dispatch Timing**
Avoid starting long-haul trips between 18:00–22:00 (peak congestion window). Batch bulk dispatches to the 06:00–10:00 window for best on-time performance.

**5. Outlier Monitoring**
Implement real-time alerting for trips that breach the IQR upper bound on delivery time. Top 5% longest trips require a separate handling SOP and dedicated operations review.

## Author

**Asmita Rajendra**
[LinkedIn](https://www.linkedin.com/in/asmita-r-5b23691a1/)· [GitHub](https://github.com/asmi2604)

*Built as a logistics data engineering and EDA case study — Delhivery, April 2026*
