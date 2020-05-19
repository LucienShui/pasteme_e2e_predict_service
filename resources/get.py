import os


def get(file_path: str, mode: str = 'r', *args, **kwargs):
    real_file_path = os.path.join(os.path.dirname(__file__), '../static', file_path)
    return open(real_file_path, mode, *args, **kwargs)
