import unittest
from src.core.local_model import LocalModel


class TestLocalModel(unittest.TestCase):
    def setUp(self):
        self.model = LocalModel()

    def test_generate_response(self):
        response = self.model.generate_response("自我介绍")
        self.assertIn("AI Nirvana", response)

    def test_generate_response_unknown(self):
        response = self.model.generate_response("未知问题")
        self.assertEqual(response, self.model.responses["默认回答"])

    def test_summarize(self):
        text = "This is a long text that needs to be summarized." * 10
        summary = self.model.summarize(text)
        self.assertLess(len(summary), len(text))
        self.assertTrue(summary.endswith("..."))


if __name__ == "__main__":
    unittest.main()
