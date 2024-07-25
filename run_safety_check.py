import sys
import subprocess


def read_requirements(file_path):
    with open(file_path, "r", encoding="utf-16") as f:
        return f.read()


def run_safety_check():
    try:
        reqs = read_requirements("requirements.txt")
        result = subprocess.run(
            ["safety", "check", "--stdin"],
            input=reqs,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return result.returncode
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_safety_check())
