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
