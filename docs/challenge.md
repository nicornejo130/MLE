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

---

## Part II: API

The API was implemented in `challenge/api.py` using FastAPI.

The model is trained on the first prediction request and cached in memory afterward. This avoids retraining on every request while keeping the implementation simple and aligned with the challenge structure.

The `/predict` endpoint validates the three raw fields required by the model:

```text
OPERA
TIPOVUELO
MES
```

Validation rules:

```text
OPERA must exist in the training data.
TIPOVUELO must be either I or N.
MES must be an integer between 1 and 12.
```

Invalid inputs return HTTP 400 responses. Valid inputs return predictions in the expected format:

```json
{"predict": [0, 1]}
```

---

## Test Compatibility

`tests/model/test_model.py` loads the dataset using a path relative to the current working directory:

```python
self.data = pd.read_csv(filepath_or_buffer="../data/data.csv")
```

This path only resolves when pytest runs from inside `tests/`, while `make model-test` runs from the repository root.

To keep both the tests and the Makefile untouched, a minimal `tests/conftest.py` was added to run the test session from the expected working directory.

---

## Part III: Deployment

The API was deployed to Google Cloud Run using a containerized FastAPI service.

Cloud Run was selected because it provides a managed deployment target for HTTP APIs, supports public HTTPS endpoints, and allows request-based billing with autoscaling.

The service was configured with:

```text
Authentication: public access
Billing: request-based
Scaling: autoscaling
Minimum instances: 0
Maximum instances: 2
Ingress: all
```

The deployed URL was added to the `STRESS_URL` variable in the Makefile.

### Stress Test

The stress test uses the Locust 2.x API:

```python
from locust import HttpUser, task
```

The original Locust dependency was missing or stale, and older Locust versions are incompatible with current Jinja2 releases. Since the stress test already uses the modern API, Locust was added explicitly as a 2.x dependency:

```text
locust~=2.31
```

`make stress-test` was executed against the deployed Cloud Run service and completed successfully with 0 failed requests.
