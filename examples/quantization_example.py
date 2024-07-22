# examples/quantization_example.py

from src.core.quantization import Quantizer
import numpy as np

def main():
    # 创建一个模拟的模型权重
    weights = np.random.randn(1000).astype(np.float32)
    
    quantizer = Quantizer()

    print("Original weights statistics:")
    print(f"Mean: {np.mean(weights):.4f}")
    print(f"Std: {np.std(weights):.4f}")
    print(f"Min: {np.min(weights):.4f}")
    print(f"Max: {np.max(weights):.4f}")
    print(f"Size: {weights.nbytes} bytes")
    print()

    # 应用量化
    quantized_weights = quantizer.quantize(weights, bits=8)

    print("Quantized weights statistics:")
    print(f"Mean: {np.mean(quantized_weights):.4f}")
    print(f"Std: {np.std(quantized_weights):.4f}")
    print(f"Min: {np.min(quantized_weights):.4f}")
    print(f"Max: {np.max(quantized_weights):.4f}")
    print(f"Size: {quantized_weights.nbytes} bytes")
    print()

    # 反量化
    dequantized_weights = quantizer.dequantize(quantized_weights)

    print("Dequantized weights statistics:")
    print(f"Mean: {np.mean(dequantized_weights):.4f}")
    print(f"Std: {np.std(dequantized_weights):.4f}")
    print(f"Min: {np.min(dequantized_weights):.4f}")
    print(f"Max: {np.max(dequantized_weights):.4f}")
    print(f"Size: {dequantized_weights.nbytes} bytes")

    # 计算量化误差
    mse = np.mean((weights - dequantized_weights) ** 2)
    print(f"\nMean Squared Error: {mse:.8f}")

if __name__ == "__main__":
    main()