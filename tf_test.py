# tf_test.py
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")