import typing
import json
from flask import Flask, request, jsonify, abort
from model_loader import load_model
from model.baseModel import BaseModel

app = Flask(__name__)

model_instances: typing.Dict[str, dict] = {}


def load_config(config_path: str):
    global model_instances
    with open(config_path) as json_config:
        config: dict = json.load(json_config)

        loaded_model: typing.Dict[str, BaseModel] = load_model(config)

        for key, value in config.items():
            model_name = key
            model_instances[model_name] = {
                'host': value['host'],
                'instance': loaded_model[key]
            }


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/v<version>/models/<model_name>:predict', methods=['POST'])
def predict(version: int, model_name: str):
    data: dict = json.loads(request.get_data().decode('utf-8'))

    if not data or 'content' not in data:
        abort(400)

    model_config: dict = model_instances[model_name]
    model: BaseModel = model_config['instance']

    content_list: typing.List[str] = data['content']

    if not isinstance(data['content'], list) and isinstance(data['content'], str):
        content_list = [data['content']]

    return jsonify(
        model.predict(
            model_config['host'], model_name, {'content': content_list}, version
        )
    )


if __name__ == '__main__':
    load_config('config.json')
    app.run()
