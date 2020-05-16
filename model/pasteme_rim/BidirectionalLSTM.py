from model import BaseModel


class BidirectionalLSTM(BaseModel):
    def preprocess(self, raw_data) -> dict:
        pass

    def after_prediction(self, prediction: dict) -> dict:
        pass
