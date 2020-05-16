import logging
import requests
import json


class BaseModel:

    def __init__(self, host: str, model_name: str = 'anonymous', version: int = 1):
        self.logger = logging.getLogger(model_name)
        self.model_name = model_name
        self.version = version
        self.host = host

    def preprocess(self, raw_data) -> dict:
        raise NotImplementedError

    def after_prediction(self, prediction: dict) -> dict:
        raise NotImplementedError

    def inference(self, raw_data: dict) -> dict:
        self.logger.info('start inference, raw_data = {}'.format(raw_data))
        http_response = requests.post(
            '{}/v{}/models/{}'.format(self.host, self.version, self.model_name),
            json=self.preprocess(raw_data)
        )
        self.logger.info('http_response = {}'.format(http_response))
        try:
            json_response = http_response.json()
            return self.after_prediction(json_response)
        except json.decoder.JSONDecodeError:
            return {
                'error': 'From pasteme_model_preprocess_service: json.decoder.JSONDecodeError',
                'http_response': http_response.content
            }
