import sys
import os

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name}")
        return True
    except ImportError as e:
        print(f"[FAIL] {module_name}: {e}")
        return False

print("Check Setup Script v1.0")
print(f"Using Python: {sys.executable}")
print("-" * 30)

modules_to_check = [
    "pandas",
    "numpy", 
    "torch",
    "transformers",
    "pydantic_settings",
    "sklearn"
]

failed = []
for m in modules_to_check:
    if not check_import(m):
        failed.append(m)

# Check internal modules
cwd = os.getcwd()
sys.path.append(cwd)
print(f"Added {cwd} to sys.path")

try:
    import backend.core.config
    print("[OK] backend.core.config import successful")
except ImportError as e:
    print(f"[FAIL] backend package import: {e}")
    failed.append("backend")

print("-" * 30)
if failed:
    print(f"XXX FAIL: The following modules are missing: {failed}")
    print("Please use the 'backend/venv' interpreter.")
    sys.exit(1)
else:
    print(">>> SUCCESS: All critical imports work correctly! <<<")
    sys.exit(0)
