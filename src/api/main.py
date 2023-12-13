import argparse

import pandas as pd
import uvicorn
from fastapi import FastAPI

from src.api.api_input import ApiInput
from src.api.api_output import ApiOutput
from src.data.utils import load_config
from src.model.random_forest_classifier import RandomForestClassifier
from src.preprocess.api_preprocess import ApiPreprocess

app = FastAPI()


@app.post("/predict/", response_model=ApiOutput, status_code=200)
async def create_item(m_input: ApiInput):
    data = api_preprocess.preprocess(pd.DataFrame([m_input.__dict__]), with_label=False)
    pred = model.predict(data)[0]

    return {
        "id": m_input.id,
        "price_category": price_to_str[pred]
    }


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--config', dest='config', required=True)
    args = argument_parser.parse_args()

    config = load_config(args.config)
    api_preprocess = ApiPreprocess(**config["api_preprocess"])
    model = RandomForestClassifier.from_pickle(model_path=config["api"]["model_path"])
    price_to_str = config["api"]["output_map"]
    uvicorn.run(app, host=config["api"]["ip"], port=config["api"]["port"])
