import sys
print(sys.path)
try:
    from src.core.security import SecurityManager
    print("Successfully imported SecurityManager")
except ImportError as e:
    print(f"Import error: {e}")