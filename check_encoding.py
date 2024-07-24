import chardet

with open('requirements.txt', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    print(f"Detected encoding: {result['encoding']}")

print("First few bytes:")
print(raw_data[:20])  # 打印前20个字节