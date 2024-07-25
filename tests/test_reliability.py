import unittest
from unittest.mock import patch
from src.core.reliability import setup_logging, handle_exception


class TestReliability(unittest.TestCase):
    @patch("logging.basicConfig")
    def test_setup_logging(self, mock_basic_config):
        class MockConfig:
            def get(self, key):
                return "INFO"

        config = MockConfig()
        setup_logging(config)
        mock_basic_config.assert_called_once()

    @patch("logging.error")
    def test_handle_exception(self, mock_error):
        @handle_exception
        def raise_exception():
            raise ValueError("Test error")

        result = raise_exception()
        self.assertEqual(
            result, "I'm sorry, an error occurred. Please try again later."
        )
        mock_error.assert_called_once_with("An error occurred: Test error")


if __name__ == "__main__":
    unittest.main()
