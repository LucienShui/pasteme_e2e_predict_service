import logging
import requests
import json


class BaseModel:

    def __init__(self):
        self.__logger__ = logging.getLogger(self.__class__.__name__)
        self.__logger__.setLevel(logging.INFO)
        pass

    def __preprocess__(self, raw_data: dict) -> dict:
        raise NotImplementedError

    def __after_prediction__(self, predict_result: dict) -> dict:
        raise NotImplementedError

    def predict(self, host: str, model_name: str, raw_data: dict) -> dict:
        http_response = requests.post(
            '{}/v1/models/{}:predict'.format(host, model_name),
            json=self.__preprocess__(raw_data)
        )
        try:
            json_response: dict = http_response.json()
            return self.__after_prediction__(json_response)

        except json.decoder.JSONDecodeError:
            return {
                'error': 'From pasteme_model_preprocess_service: json.decoder.JSONDecodeError',
                'http_response': http_response.content
            }
