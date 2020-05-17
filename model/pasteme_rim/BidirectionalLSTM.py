import json
import typing
import pandas as pd
from model import BaseModel
from model.pasteme_rim import preprocess


class BidirectionalLSTM(BaseModel):

    def __init__(self, host: str, model_name: str = 'anonymous', version: int = 1, max_length: int = 128):
        super().__init__(host, model_name, version)
        self.word2idx: typing.Dict[str, int] = json.load(open('model/pasteme_rim/word2idx.json'))
        self.max_length = max_length

    def preprocess(self, raw_data: dict) -> dict:
        df = pd.DataFrame(columns=['text'], data=raw_data['content'])
        df = preprocess.extract_chinese(df)
        df = preprocess.tokenize(df)
        df = preprocess.tokens_to_ids(df, self.word2idx)
        result = preprocess.pad_sequences(df['tokens'].values)
        return {'instances': result.tolist()}

    def after_prediction(self, predict_result: dict) -> dict:
        predictions = predict_result['predictions']
        result = []
        for each in predictions:
            result.append('normal' if each == 0 else 'risk')

        return {'result': result}
