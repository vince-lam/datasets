# TabH2O workshop datasets

Small, pre-engineered train/test splits used by the TabH2O hands-on workshop
notebook. Every file is a **sample** of a larger public dataset, reduced to fit
comfortably inside the TabH2O API request limits (100k rows, 100 columns, 50 MB)
and to keep a live workshop responsive.

Load them directly over HTTP:

```python
import pandas as pd

BASE = "https://raw.githubusercontent.com/vince-lam/datasets/main/data/tabh2o_workshop"
train = pd.read_csv(f"{BASE}/hdb_train.csv")
```

## Files

| File | Rows | Cols | Task | Target / label |
|---|---:|---:|---|---|
| `hdb_train.csv` | 8,000 | 12 | regression | `resale_price` |
| `hdb_test.csv` | 1,000 | 12 | regression | `resale_price` (ground truth, withheld from the request) |
| `fraud_train.csv` | 8,000 | 21 | classification | `is_fraud` (~5.5% positive) |
| `fraud_test.csv` | 2,000 | 21 | classification | `is_fraud` (ground truth) |
| `atm_train.csv` | 11,375 | 13 | forecasting | `total_amount_withdrawn`, time `transaction_date` |
| `atm_test.csv` | 139 | 13 | forecasting | 28-day horizon × 5 ATMs (ground truth) |
| `cards.csv` | 8,950 | 18 | clustering + imputation | none — 314 genuinely missing cells |
| `aml.csv` | 20,000 | 18 | anomaly detection | `Is Laundering` (~0.05% positive, withheld) |

The `_test.csv` files keep the target column so the notebook can score
predictions against ground truth. The target is dropped before the API request.

## Sources and attribution

These are derived samples, not raw redistributions. Original sources:

| Dataset | Source | Licence |
|---|---|---|
| HDB resale prices | [data.gov.sg](https://data.gov.sg/collections/189/view) — 236k rows | Singapore Open Data Licence |
| Bank transaction fraud | [Kaggle](https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset) — 1M rows | see source page |
| ATM withdrawals (XYZ bank) | [Kaggle](https://www.kaggle.com/datasets/saadfareed/data-set-of-atm-transaction-of-xyz-bank) — 5 ATMs × 2,427 days | see source page |
| Credit card usage | [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) — 8,950 customers | see source page |
| IBM AML transactions | [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — LI-Small, 6.9M rows | see source page |

Feature engineering (calendar features, trailing level anchors, log transforms,
entity counts) was applied before sampling — see the workshop notebook for what
each column means.
