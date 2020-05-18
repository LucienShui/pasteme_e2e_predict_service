import unittest
from model.pasteme_rim import BidirectionalLSTM


class MyModelTestCase(unittest.TestCase):
    def test_BidirectionalLSTM(self):
        model = BidirectionalLSTM(
            host='http://docker:8501',
            model_name='PasteMeRIM',
            version=1, max_length=128)
        prediction = model.predict({'content': ['你好，世界！']})
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
