import unittest
import json
from .get import get


class MyGetTestCase(unittest.TestCase):
    def test_something(self):
        with get('config.json') as file:
            print(json.load(file))
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
