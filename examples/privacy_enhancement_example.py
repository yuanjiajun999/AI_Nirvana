# examples/privacy_enhancement_example.py

from src.core.privacy_enhancement import PrivacyEnhancement

def main():
    privacy_enhancer = PrivacyEnhancement(epsilon=1.0)

    # 差分隐私示例
    sensitive_data = [10, 20, 30, 40, 50]
    print("Original Data:", sensitive_data)
    
    private_data = privacy_enhancer.apply_differential_privacy(sensitive_data)
    print("Data with Differential Privacy:", private_data)

    # 加密敏感数据示例
    sensitive_info = "This is sensitive information"
    print("\nOriginal Sensitive Info:", sensitive_info)
    
    encrypted_info = privacy_enhancer.encrypt_sensitive_data(sensitive_info)
    print("Encrypted Info:", encrypted_info)
    
    decrypted_info = privacy_enhancer.decrypt_sensitive_data(encrypted_info)
    print("Decrypted Info:", decrypted_info)

    # 设置新的隐私预算
    privacy_enhancer.set_privacy_budget(epsilon=0.5)
    print("\nUpdated privacy budget (epsilon) to 0.5")
    
    new_private_data = privacy_enhancer.apply_differential_privacy(sensitive_data)
    print("Data with New Privacy Budget:", new_private_data)

if __name__ == "__main__":
    main()