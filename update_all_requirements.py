import subprocess
import re

def is_platform_specific(req):
    platform_specific = ['pywin32', 'pywinpty', 'win32', 'wincertstore']
    return any(pkg in req for pkg in platform_specific)

def update_all_requirements():
    # 更新所有依赖
    print("Updating all dependencies...")
    subprocess.run(["pip", "install", "--upgrade", "-r", "requirements.txt"])
    
    # 生成新的 requirements.txt
    print("Generating new requirements.txt...")
    subprocess.run(["pip", "freeze"], stdout=open("requirements.txt", "w"))
    
    # 读取更新后的 requirements.txt
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    
    # 过滤平台特定的依赖
    common_reqs = [req for req in requirements if not is_platform_specific(req)]
    
    # 写入 requirements-common.txt
    print("Updating requirements-common.txt...")
    with open("requirements-common.txt", "w") as f:
        f.write("\n".join(common_reqs))
    
    # 写入 requirements-docker.txt
    print("Updating requirements-docker.txt...")
    with open("requirements-docker.txt", "w") as f:
        f.write("\n".join(common_reqs))
    
    print("All requirements files have been updated successfully.")

if __name__ == "__main__":
    update_all_requirements()