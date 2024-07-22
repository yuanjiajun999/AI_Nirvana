import subprocess
import argparse
from importlib.metadata import version, PackageNotFoundError

def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

def get_latest_version(package_name):
    try:
        result = subprocess.run(['pip', 'install', f'{package_name}==', '--dry-run'], capture_output=True, text=True)
        output = result.stderr
        latest_version = output.split('from versions: ')[-1].strip()
        return latest_version
    except Exception:
        return None

def parse_requirement(req):
    parts = req.split('==')
    return parts[0], parts[1] if len(parts) > 1 else None

def install_package(package, upgrade=False):
    cmd = ["pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    try:
        subprocess.check_call(cmd)
        print(f"Successfully installed/upgraded {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Install or upgrade Python packages.")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade packages to the specified version")
    parser.add_argument("--skip-installed", action="store_true", help="Skip already installed packages")
    args = parser.parse_args()

    dependencies = [
        "absl-py==2.1.0",
        "aiohttp==3.9.5",
        "aiosignal==1.3.1",
        "requests==2.26.0",  # 新添加的包用于测试
        # ... 添加所有依赖项
    ]

    for dependency in dependencies:
        package_name, required_version = parse_requirement(dependency)
        installed_version = get_installed_version(package_name)

        if installed_version:
            if args.skip_installed and not args.upgrade:
                print(f"Skipping {package_name} (already installed)")
                continue
            elif installed_version == required_version and not args.upgrade:
                print(f"Skipping {package_name} (version {installed_version} already satisfied)")
                continue
            elif args.upgrade:
                latest_version = get_latest_version(package_name)
                if latest_version and latest_version != installed_version:
                    print(f"Upgrading {package_name} from {installed_version} to {latest_version}")
                    install_package(f"{package_name}=={latest_version}", upgrade=True)
                else:
                    print(f"Skipping {package_name} (already up to date)")
            else:
                print(f"Installing {package_name} {required_version} (current: {installed_version})")
                install_package(dependency)
        else:
            print(f"Installing {package_name} {required_version}")
            install_package(dependency)

if __name__ == "__main__":
    main()