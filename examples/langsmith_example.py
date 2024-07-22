# examples/langsmith_example.py

from src.core.langsmith import LangSmith

def main():
    lang_smith = LangSmith()

    # 代码生成示例
    prompt = "Write a Python function to calculate the factorial of a number"
    generated_code = lang_smith.generate_code(prompt)
    print("Generated Code:")
    print(generated_code)
    print()

    # 代码重构示例
    code_to_refactor = """
    def f(x):
        if x == 0:
            return 1
        else:
            return x * f(x-1)
    """
    refactored_code = lang_smith.refactor_code(code_to_refactor)
    print("Original Code:")
    print(code_to_refactor)
    print("Refactored Code:")
    print(refactored_code)
    print()

    # 文本翻译示例
    text = "Hello, world!"
    target_lang = "French"
    translated_text = lang_smith.translate_text(text, target_lang)
    print(f"Original text: {text}")
    print(f"Translated to {target_lang}: {translated_text}")

if __name__ == "__main__":
    main()