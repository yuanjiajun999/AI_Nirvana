import tensorflow as tf  
import keras  
import h5py  
import numpy as np  

def test_imports():  
    print("Testing imports:")  
    print(f"TensorFlow version: {tf.__version__}")  
    print(f"Keras version: {keras.__version__}")  
    print(f"h5py version: {h5py.__version__}")  
    print(f"NumPy version: {np.__version__}")  

def test_gpu():  
    print("\nTesting GPU availability:")  
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  
    print("Is GPU available: ", tf.test.is_gpu_available())  

def test_tensorflow():  
    print("\nTesting TensorFlow:")  
    # 使用相同的数据类型（浮点数）  
    x = tf.constant([[1., 2.], [3., 4.]])  
    y = tf.constant([[1., 2.], [3., 4.]])  # 改为浮点数  
    z = tf.matmul(x, y)  
    print("TensorFlow matrix multiplication result:")  
    print(z.numpy())

if __name__ == "__main__":  
    test_imports()  
    test_gpu()  
    test_tensorflow()