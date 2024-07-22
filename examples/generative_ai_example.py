# examples/generative_ai_example.py

from src.core.generative_ai import GenerativeAI

def main():
    gen_ai = GenerativeAI()

    # 文本生成示例
    prompt = "In a world where AI has become ubiquitous,"
    generated_text = gen_ai.generate_text(prompt, max_length=100)
    print("Generated Text:")
    print(generated_text)

    # 图像生成示例
    image_prompt = "A futuristic city with flying cars and tall skyscrapers"
    image = gen_ai.generate_image(image_prompt)
    print("\nImage generated. Saving as 'generated_image.png'")
    image.save("generated_image.png")

    # 代码生成示例
    code_prompt = "Create a Python function to calculate the Fibonacci sequence"
    generated_code = gen_ai.generate_code(code_prompt)
    print("\nGenerated Code:")
    print(generated_code)

if __name__ == "__main__":
    main()