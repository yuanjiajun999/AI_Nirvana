import os
import subprocess


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, error = process.communicate()
    return output.decode(), error.decode()


def fix_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Fixing {file_path}")

                # Run isort
                run_command(f"isort {file_path}")

                # Run autopep8
                run_command(
                    f"autopep8 --in-place --aggressive --aggressive {file_path}"
                )


if __name__ == "__main__":
    directories = ["src", "tests", "examples"]
    for directory in directories:
        fix_directory(directory)
    print("Auto-fixing complete. Please review the changes and run flake8 again.")
