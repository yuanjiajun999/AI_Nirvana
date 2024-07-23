# update_requirements.py
import pkg_resources
import re

def is_platform_specific(req):
    platform_specific = ['pywin32', 'pywinpty', 'win32', 'wincertstore']
    return any(pkg in req for pkg in platform_specific)

def update_common_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()

    common_reqs = [req for req in requirements if not is_platform_specific(req)]

    with open('requirements-common.txt', 'w') as f:
        f.write('\n'.join(common_reqs))

if __name__ == "__main__":
    update_common_requirements()
    print("requirements-common.txt has been updated.")