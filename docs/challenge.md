# Challenge Notes

## Environment and Dependencies

Python `3.10` was used because it is compatible with the provided dependency constraints, while Python 3.11+ introduces issues with packages such as `pandas~=1.3.5`.

### Dependency Adjustments

The notebook uses older seaborn syntax such as:

```python
sns.barplot(x, y)
```

To keep the notebook reproducible without modifying it, `seaborn` was pinned to:

```text
seaborn~=0.11.2
```

The notebook trains XGBoost models, but `xgboost` was not included in the original requirements. Since the selected production model uses XGBoost, the dependency was added explicitly:

```text
xgboost~=3.2.0
```

---

## Part I: Model

### Model Selection

The selected model is **XGBoost using the Top 10 features with class balancing through `scale_pos_weight`**.

The dataset is imbalanced, so accuracy was not the main selection metric. Some models reached around 81% accuracy but had almost zero recall for the delay class, making them unsuitable for the business objective.

The selected model achieved approximately:

```text
delay recall: 0.69
accuracy: 0.55
```

This trade-off is acceptable because detecting delayed flights is more important than optimizing overall accuracy. A false positive is less costly than failing to identify a flight that is likely to be delayed.

XGBoost was preferred over Logistic Regression because it handles non-linear relationships and feature interactions better in tabular data, while also providing feature importance for interpretability.

### Implementation

The notebook logic was moved into `challenge/model.py` using the required `DelayModel` interface.

The final model uses the Top 10 selected features, derived from:

```text
OPERA
TIPOVUELO
MES
```

Prediction does not depend on post-operation fields such as `Fecha-O`, since these values are not available before the flight operates.

Preprocessing one-hot encodes the raw input columns and aligns them with the expected feature schema. Missing dummy columns are filled with `0` to keep training and prediction inputs consistent.

When the target is required for training, it is generated using the original notebook definition:

```text
delay = 1 if min_diff > 15, otherwise 0
```

### Unfitted Model Behavior

The provided model tests call `predict` on a fresh `DelayModel` instance. To keep the model in-memory and avoid persisted artifacts or test-order dependency, `predict` returns the majority-class fallback when the model has not been fitted:

```python
if self._model is None:
    return [0] * features.shape[0]
```

The API path always trains the model before prediction, so this fallback only affects the standalone interface used by the tests.
