import unittest
from src.core.quantization import prepare_model_for_quantization, quantize_and_evaluate

class TestQuantization(unittest.TestCase):
    # Test cases for quantization implementation