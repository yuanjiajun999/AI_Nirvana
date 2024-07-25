import os
import re


def fix_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    fixed_lines = []
    for line in lines:
        # 删除行尾空白字符
        line = line.rstrip() + "\n"

        # 如果行长度超过79个字符，尝试在79个字符处断行
        if len(line.rstrip()) > 79:
            words = line.split()
            new_line = ""
            for word in words:
                if len(new_line) + len(word) > 79:
                    fixed_lines.append(new_line.rstrip() + "\n")
                    new_line = "    " + word + " "  # 添加4个空格的缩进
                else:
                    new_line += word + " "
            if new_line:
                fixed_lines.append(new_line.rstrip() + "\n")
        else:
            fixed_lines.append(line)

    # 确保文件末尾有一个空行
    if fixed_lines and not fixed_lines[-1].isspace():
        fixed_lines.append("\n")

    # 修复空行问题
    final_lines = []
    for i, line in enumerate(fixed_lines):
        if i > 0 and line.strip() and fixed_lines[i - 1].strip():
            if re.match(r"^(class|def)\s", line):
                final_lines.append("\n\n")
        final_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(final_lines)


def fix_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                fix_file(os.path.join(root, file))


if __name__ == "__main__":
    directories = ["src", "tests", "examples"]
    for directory in directories:
        fix_directory(directory)
    print("Code style fixes applied.")
