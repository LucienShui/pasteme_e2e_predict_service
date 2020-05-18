import typing
from model.baseModel import BaseModel
from model.pasteme_rim import BidirectionalLSTM


def load_model(config: dict) -> typing.Dict[str, BaseModel]:
    loaded_model = {
        'PasteMeRIM': BidirectionalLSTM(config['PasteMeRIM']['max_length'])
    }
    return loaded_model
