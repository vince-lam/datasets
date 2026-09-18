# CMLP programme datasets

Datasets for the Certified Machine Learning Practitioner (CMLP) advanced
training programme. Two banking problems, chosen because they fail in opposite
ways:

- **Bank Account Fraud (BAF)** - extreme class imbalance and temporal drift. A
  time-ordered train/test split, so the test months come strictly after the
  training months.
- **Lending Club** - target leakage. The file ships with ten columns recorded
  *after* the loan outcome was known. A model trained on them scores beautifully
  and is worth nothing.

Load them directly over HTTP:

```python
import pandas as pd

BASE = "https://raw.githubusercontent.com/vince-lam/datasets/main/data/cmlp-programme"
train = pd.read_csv(f"{BASE}/baf_base_train_m0-5.csv")
test  = pd.read_csv(f"{BASE}/baf_base_test_m6-7.csv")
loans = pd.read_csv(f"{BASE}/lendingclub_leakage_demo.csv", low_memory=False)
```

## Files

| File | Rows | Cols | Task | Target | Positive rate |
|---|---:|---:|---|---|---:|
| `baf_base_train_m0-5.csv` | 119,248 | 32 | classification | `fraud_bool` | 1.080% (1,288) |
| `baf_base_test_m6-7.csv` | 30,751 | 32 | classification | `fraud_bool` | 1.434% (441) |
| `lendingclub_leakage_demo.csv` | 38,577 | 41 | classification | `default_bool` | 14.59% (5,627) |
| `lendingclub_data_dictionary.xlsx` | - | - | reference | - | - |

## Bank Account Fraud

Sampled from the BAF `Base` variant. Split on the `month` column, which runs
0-7 in the source: **months 0-5 train, months 6-7 test**. The split is
deliberately temporal rather than random, so the drift between the two periods
is real and visible - the fraud rate rises from 1.08% to 1.43% across the
boundary.

`month` is kept in both files so the drift can be inspected. Drop it before
training.

## Lending Club

2007-2011 vintages, `issue_d` from Apr-08 to Sep-11. The leakage columns:

| Column | When it is known |
|---|---|
| `total_pymnt`, `total_pymnt_inv` | after repayment |
| `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | after repayment |
| `recoveries`, `collection_recovery_fee` | **only if the loan defaulted** |
| `last_pymnt_d`, `last_pymnt_amnt` | after the last payment |
| `last_credit_pull_d` | after servicing |

These are not synthetic plants - they are the kind of column a bank's own
feature store hands you when someone joins the servicing table to the
origination table. `id` and `member_id` are identifiers and should also be
dropped.

`lendingclub_data_dictionary.xlsx` is the original column documentation. Working
out which columns are post-outcome *from the documentation* is part of the
exercise.

## Sources and attribution

These are derived samples, not raw redistributions.

| Dataset | Source | Licence |
|---|---|---|
| Bank Account Fraud | [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) - Jesus et al., NeurIPS 2022 Datasets and Benchmarks track (Feedzai), `Base` variant, 1M rows | CC BY-NC-SA 4.0 |
| Lending Club | [Kaggle](https://www.kaggle.com/datasets/wendykan/lending-club-loan-data) - 2007-2011 accepted loans | see source page |

Bank Account Fraud is licensed **CC BY-NC-SA 4.0**, which is non-commercial. The
sample here is redistributed for training and educational use under that
licence; check the source terms before any commercial use. The repository's MIT
licence covers the repository, not the third-party data in this directory.
