import logging
import requests
import json


class BaseModel:

    def __init__(self, host: str, model_name: str = 'anonymous', version: int = 1):
        self.logger = logging.getLogger(model_name)
        self.host: str = host
        self.model_name: str = model_name
        self.version: int = version

    def preprocess(self, raw_data: dict) -> dict:
        raise NotImplementedError

    def after_prediction(self, predict_result: dict) -> dict:
        raise NotImplementedError

    def predict(self, raw_data: dict) -> dict:
        self.logger.info('start inference, raw_data = {}'.format(raw_data))
        http_response = requests.post(
            '{}/v{}/models/{}:predict'.format(self.host, self.version, self.model_name),
            json=self.preprocess(raw_data)
        )
        self.logger.info('http_response = {}'.format(http_response))
        try:
            json_response: dict = http_response.json()
            return self.after_prediction(json_response)
        except json.decoder.JSONDecodeError:
            return {
                'error': 'From pasteme_model_preprocess_service: json.decoder.JSONDecodeError',
                'http_response': http_response.content
            }
