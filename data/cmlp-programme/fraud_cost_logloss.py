"""Cost-sensitive logloss for the workshop fraud dataset.
A missed fraud (FN) costs the transaction amount; a false alarm (FP) costs a flat review fee.
Adapted from h2oai/driverlessai-recipes scorers/classification/binary/logloss_with_costs.py
"""
import typing

import numpy as np
import datatable as dt
from datatable import f
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder


class FraudCostLogloss(CustomScorer):
    _description = "Logloss weighted by business costs: (fn_cost*FN + fp_cost*FP + tp_cost*TP + tn_cost*TN) / N"
    _binary = True
    _maximize = False          # lower is better
    _perfect_score = 0
    _display_name = "FraudCostLogloss"   # the name you will see in the Scorer list
    _needs_X = True            # DAI passes the ORIGINAL dataset so we can read the cost column
    _epsilon = 1e-15

    # ------------------------------------------------------------------
    # EDIT THESE FOUR FOR YOUR DATASET
    # A string is a column name in the original dataset (per-row cost).
    # A number is a constant cost for every row.
    # ------------------------------------------------------------------
    _fn_cost = 'amount'   # missed fraud: we lose the transaction amount
    _fp_cost = 5.0        # false alarm: flat cost of a review / customer contact
    _tp_cost = 0.0        # caught fraud: set to the review cost (e.g. 2.0) if reviews cost money
    _tn_cost = 0.0        # correct approval: free

    # Cap for per-row costs so a handful of very large transactions do not
    # dominate the score. Set to None to disable.
    _cost_cap = 2000.0
    # ------------------------------------------------------------------

    def make_cost_values(self, cost_value, X, shape, default_value):
        """Column name -> per-row costs from X; number -> constant. Falls back to
        default_value when the column is not in X (this is what lets the recipe
        pass the acceptance test on DAI's synthetic data)."""
        if isinstance(cost_value, str):
            if isinstance(X, dt.Frame) and cost_value in X.names:
                # to_list handles missing values (None -> nan); then fill and cap
                cost = np.array(X[:, cost_value].to_list()[0], dtype=float)
                cost = np.where(np.isnan(cost), default_value, cost)
                if self.__class__._cost_cap is not None:
                    cost = np.minimum(cost, self.__class__._cost_cap)
            else:
                cost = np.full(shape, default_value, dtype=float)
        elif isinstance(cost_value, (int, float)):
            cost = np.full(shape, float(cost_value))
        else:
            raise ValueError("Cost must be a column name (str) or a number.")
        return cost

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              X: typing.Optional[dt.Frame] = None,
              **kwargs) -> float:
        N = actual.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(N)

        lb = LabelEncoder()
        labels = lb.fit_transform(labels)

        DT = dt.Frame(actual=lb.transform(actual),
                      predicted=np.minimum(1 - self.__class__._epsilon,
                                           np.maximum(self.__class__._epsilon, predicted)),
                      cost_fn=self.make_cost_values(self.__class__._fn_cost, X, N, 1.),
                      cost_tp=self.make_cost_values(self.__class__._tp_cost, X, N, 0.),
                      cost_tn=self.make_cost_values(self.__class__._tn_cost, X, N, 0.),
                      cost_fp=self.make_cost_values(self.__class__._fp_cost, X, N, 1.),
                      sample_weight=sample_weight)

        # fraud rows:   cost_fn * log(p)      punishes a low probability on a real fraud
        #               cost_tp * log(1 - p)  punishes a high probability (only if reviews cost money)
        # genuine rows: cost_fp * log(1 - p)  punishes a high probability on a genuine transaction
        #               cost_tn * log(p)      normally zero
        lloss = DT[:, f.sample_weight * (f.actual * (f.cost_fn * dt.log(f.predicted) +
                                                     f.cost_tp * dt.log(1 - f.predicted)) +
                                         (1 - f.actual) * (f.cost_fp * dt.log(1 - f.predicted) +
                                                           f.cost_tn * dt.log(f.predicted)))]
        return lloss.sum()[0, 0] * -1.0 / np.sum(sample_weight)
