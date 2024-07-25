import subprocess
import argparse
from importlib.metadata import version, PackageNotFoundError
import os


def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def get_latest_version(package_name):
    try:
        result = subprocess.run(
            ["pip", "install", f"{package_name}==", "--dry-run"],
            capture_output=True,
            text=True,
        )
        output = result.stderr
        latest_version = output.split("from versions: ")[-1].strip()
        return latest_version
    except Exception:
        return None


def parse_requirement(req):
    parts = req.split("==")
    return parts[0], parts[1] if len(parts) > 1 else None


def install_package(package, upgrade=False, max_retries=3):
    cmd = ["pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)

    for attempt in range(max_retries):
        try:
            subprocess.check_call(cmd)
            print(f"Successfully installed/upgraded {package}")
            return
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed to install {package}: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to install {package} after {max_retries} attempts")


def read_requirements(file_path):
    with open(file_path, "r") as file:
        return [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]


def main():
    parser = argparse.ArgumentParser(description="Install or upgrade Python packages.")
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade packages to the specified version",
    )
    parser.add_argument(
        "--skip-installed", action="store_true", help="Skip already installed packages"
    )
    parser.add_argument(
        "--requirements-file",
        default="requirements-docker.txt",
        help="Path to the requirements file",
    )
    args = parser.parse_args()

    # 在 Docker 环境中使用 requirements-docker.txt
    if os.environ.get("DOCKER_ENV") == "1":
        args.requirements_file = "requirements-docker.txt"
        print("Docker environment detected. Using requirements-docker.txt.")

    # 如果指定的文件不存在，回退到使用 requirements.txt
    if not os.path.exists(args.requirements_file):
        args.requirements_file = "requirements.txt"
        print(
            f"Using {args.requirements_file} as the specified requirements file does not exist."
        )

    dependencies = read_requirements(args.requirements_file)

    for dependency in dependencies:
        package_name, required_version = parse_requirement(dependency)
        installed_version = get_installed_version(package_name)

        if installed_version:
            if args.skip_installed and not args.upgrade:
                print(f"Skipping {package_name} (already installed)")
                continue
            elif installed_version == required_version and not args.upgrade:
                print(
                    f"Skipping {package_name} (version {installed_version} already satisfied)"
                )
                continue
            elif args.upgrade:
                latest_version = get_latest_version(package_name)
                if latest_version and latest_version != installed_version:
                    print(
                        f"Upgrading {package_name} from {installed_version} to {latest_version}"
                    )
                    install_package(f"{package_name}=={latest_version}", upgrade=True)
                else:
                    print(f"Skipping {package_name} (already up to date)")
            else:
                print(
                    f"Installing {package_name} {required_version} (current: {installed_version})"
                )
                install_package(dependency)
        else:
            print(f"Installing {package_name} {required_version}")
            install_package(dependency)

    # 检查是否存在 failed-requirements.txt 并输出警告
    if os.path.exists("failed-requirements.txt"):
        print("\nWARNING: Some packages failed to install during Docker build.")
        print("Contents of failed-requirements.txt:")
        with open("failed-requirements.txt", "r") as file:
            print(file.read())


if __name__ == "__main__":
    main()
