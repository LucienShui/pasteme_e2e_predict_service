import json
import typing
# import os
import resources
import pandas as pd
from model import BaseModel
from model.pasteme_rim import preprocess


class BidirectionalLSTM(BaseModel):

    def __init__(self, max_length: int = 128):
        super().__init__()
        # word2idx_path = os.path.join(os.path.dirname(__file__), 'word2idx.json')
        # with open(word2idx_path) as file:
        with resources.get('word2idx.json') as file:
            self.word2idx: typing.Dict[str, int] = json.load(file)
        self.max_length = max_length

    def __preprocess__(self, raw_data: dict) -> dict:
        df = pd.DataFrame(columns=['text'], data=raw_data['content'])
        df = preprocess.extract_chinese(df)
        df = preprocess.tokenize(df)
        df = preprocess.tokens_to_ids(df, self.word2idx)
        result = preprocess.pad_sequences(df['text'].values, max_len=self.max_length)
        return {'instances': result.tolist()}

    def __after_prediction__(self, predict_result: dict) -> dict:
        predictions = predict_result['predictions']
        result = []
        try:
            for prediction in predictions:
                assert len(prediction) == 1
                result.append(0 if prediction[0] < 0 else 1)
        except AssertionError:
            return {'error': 'AssertionError'}

        return {'predictions': result}
