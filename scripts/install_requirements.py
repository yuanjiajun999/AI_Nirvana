import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

with open('requirements.txt', 'r', encoding='utf-8') as f:
    packages = f.read().splitlines()

for package in packages:
    if package and not package.startswith('#'):
        install(package)