# Datasets

Repo to store datasets used in workshops and tutorials.

* Datasets less than 50 MB (GitHub's file size limit) will be stored as a `csv`, `parquert`, or `zip` file.
* Datasets larger than 50 MB will be converted to a `parquet` file and split into < 50 MB parquet files.

## Datasets

|    Dataset   | Problem Type | Instances | Features | Source |
|:------------:|--------------|-----------|----------|--------|
| Wine quality | Regression   | 1599      | 11       | <https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009> |
| HDB resale prices | Regression | 9,000 (8,000 train / 1,000 test) | 12 | <https://data.gov.sg/collections/189/view> |
| Bank transaction fraud | Classification | 10,000 (8,000 train / 2,000 test) | 21 | <https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset> |
| ATM withdrawals (XYZ bank) | Forecasting | 11,514 (11,375 train / 139 test) | 13 | <https://www.kaggle.com/datasets/saadfareed/data-set-of-atm-transaction-of-xyz-bank> |
| Credit card usage | Clustering / imputation | 8,950 | 18 | <https://www.kaggle.com/datasets/arjunbhasin2013/ccdata> |
| IBM AML transactions | Anomaly detection | 20,000 | 18 | <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml> |

### Pre-split datasets

The five datasets below the wine quality row are pre-engineered samples of larger
public datasets, kept small (under 100k rows, 100 columns and 50 MB) so they load
and train quickly. Calendar features, trailing level anchors, log transforms and
entity counts were applied before sampling, so these are derived samples rather
than raw redistributions - see the original sources above for the full data and
licence terms (HDB resale prices is under the Singapore Open Data Licence).

| File | Rows | Cols | Task | Target / label |
|---|---:|---:|---|---|
| `hdb_train.csv` | 8,000 | 12 | regression | `resale_price` |
| `hdb_test.csv` | 1,000 | 12 | regression | `resale_price` (ground truth) |
| `fraud_train.csv` | 8,000 | 21 | classification | `is_fraud` (~5.5% positive) |
| `fraud_test.csv` | 2,000 | 21 | classification | `is_fraud` (ground truth) |
| `atm_train.csv` | 11,375 | 13 | forecasting | `total_amount_withdrawn`, time `transaction_date` |
| `atm_test.csv` | 139 | 13 | forecasting | 28-day horizon x 5 ATMs (ground truth) |
| `cards.csv` | 8,950 | 18 | clustering + imputation | none - 314 genuinely missing cells |
| `aml.csv` | 20,000 | 18 | anomaly detection | `Is Laundering` (~0.05% positive) |

The `_test.csv` files keep the target column so predictions can be scored against
ground truth - drop it before inference.
