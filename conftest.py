import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
root = Path(__file__).parent
sys.path.insert(0, str(root))