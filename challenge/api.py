# challenge/api.py

from functools import lru_cache
from pathlib import Path
from typing import Any

import fastapi
import pandas as pd
from fastapi import HTTPException, Request

from challenge.model import DelayModel


app = fastapi.FastAPI()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"

REQUIRED_COLUMNS = ("OPERA", "TIPOVUELO", "MES")
ALLOWED_TIPOVUELO = {"I", "N"}


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> dict:
    payload = await _read_payload(request)

    model, known_opera = _get_model()
    flights = _validate_payload(payload, known_opera)

    data = pd.DataFrame(flights)
    features = model.preprocess(data)
    predictions = model.predict(features)

    return {"predict": predictions}


@lru_cache(maxsize=1)
def _get_model() -> tuple[DelayModel, set[str]]:
    """
    Load training data, train the model once, and cache it for future requests.
    """
    data = pd.read_csv(DATA_PATH, low_memory=False)

    model = DelayModel()
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)

    known_opera = set(data["OPERA"].dropna().unique())

    return model, known_opera


async def _read_payload(request: Request) -> dict:
    """
    Read and validate that the request body is valid JSON.
    """
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Payload must be a JSON object.",
        )

    return payload


def _validate_payload(
    payload: dict[str, Any],
    known_opera: set[str],
) -> list[dict[str, Any]]:
    """
    Validate request payload and return normalized flight records.
    """
    if "flights" not in payload:
        raise HTTPException(
            status_code=400,
            detail="Payload must contain 'flights'.",
        )

    flights = payload["flights"]

    if not isinstance(flights, list) or len(flights) == 0:
        raise HTTPException(
            status_code=400,
            detail="'flights' must be a non-empty list.",
        )

    validated_flights = []

    for index, flight in enumerate(flights):
        validated_flights.append(
            _validate_flight(
                flight=flight,
                index=index,
                known_opera=known_opera,
            )
        )

    return validated_flights


def _validate_flight(
    flight: Any,
    index: int,
    known_opera: set[str],
) -> dict[str, Any]:
    """
    Validate a single flight record.
    """
    if not isinstance(flight, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Flight at index {index} must be an object.",
        )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in flight]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Flight at index {index} is missing columns: {missing_columns}.",
        )

    opera = flight["OPERA"]
    tipo_vuelo = flight["TIPOVUELO"]
    mes = flight["MES"]

    if not isinstance(opera, str) or opera not in known_opera:
        raise HTTPException(
            status_code=400,
            detail="Unknown OPERA.",
        )

    if not isinstance(tipo_vuelo, str) or tipo_vuelo not in ALLOWED_TIPOVUELO:
        raise HTTPException(
            status_code=400,
            detail="Unknown TIPOVUELO.",
        )

    if isinstance(mes, bool) or not isinstance(mes, int) or not 1 <= mes <= 12:
        raise HTTPException(
            status_code=400,
            detail="Unknown MES.",
        )

    return {
        "OPERA": opera,
        "TIPOVUELO": tipo_vuelo,
        "MES": mes,
    }
