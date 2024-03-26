from typing import List

from pydantic import BaseModel


class HealthCheck(BaseModel):
    name: str
    api_version: str
    status: str = "OK"


class UniqueMunicipalityNumber(BaseModel):
    values: List[int]


class UniqueBranch(BaseModel):
    values: List[int]


class PredictionResults(BaseModel):
    historical_datetime: List[int]
    historical_consumption: List[float]
    prediction_datetime: List[int]
    prediction_consumption: List[float]


class MonitorMetrics(BaseModel):
    datetime: List[int]
    mape: List[float]
    rmspe: List[float]


class MonitorPrediction(BaseModel):
    ground_truth_datetime: List[int]
    ground_truth_consumption: List[float]
    cached_prediction_datetime: List[int]
    cached_prediction_consumption: List[float]
