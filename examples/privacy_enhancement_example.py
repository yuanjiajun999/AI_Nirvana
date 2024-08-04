# privacy_enhancement_example.py

from src.core.privacy_enhancement import PrivacyEnhancement, AdvancedAnonymization, FederatedLearningSimulator, HomomorphicEncryptionSimulator

def main():
    # 初始化 PrivacyEnhancement 类
    pe = PrivacyEnhancement(epsilon=0.1)

    # 差分隐私示例
    original_data = [1, 2, 3, 4, 5]
    print(f"原始数据: {original_data}")
    dp_data = pe.apply_differential_privacy(original_data)
    print(f"应用差分隐私后的数据: {dp_data}")

    # 加密示例
    sensitive_info = "这是敏感信息"
    encrypted = pe.encrypt_data(sensitive_info)
    print(f"加密后的数据: {encrypted}")
    decrypted = pe.decrypt_data(encrypted)
    print(f"解密后的数据: {decrypted}")

    # k-匿名化示例
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    k = 3
    anonymized = pe.k_anonymity(data, k)
    print(f"k-匿名化后的数据 (k={k}): {anonymized}")

    # 局部差分隐私示例
    binary_data = [0, 1, 0, 1, 1]
    ldp_data = pe.local_differential_privacy(binary_data)
    print(f"应用局部差分隐私后的数据: {ldp_data}")

    # 高级匿名化示例
    aa = AdvancedAnonymization()
    aa.t_closeness(data, "sensitive_attribute", 0.1)
    aa.l_diversity(data, "sensitive_attribute", 3)

    # 联邦学习模拟示例
    fls = FederatedLearningSimulator()
    fls.simulate_federated_learning(client_data=[1, 2, 3], aggregation_function=sum)

    # 同态加密模拟示例
    hes = HomomorphicEncryptionSimulator()
    hes.simulate_homomorphic_encryption(data)

if __name__ == "__main__":
    main()