import json

from pydantic import BaseModel


class PredictionStats(BaseModel):
    fraction: float
    model_name: str
    model_type: str
    elapsed: float


def predictionstats_to_str(stats: PredictionStats):
    return json.loads(stats.json(), parse_float=str)
