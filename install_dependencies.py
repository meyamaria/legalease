# install_dependencies.py
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"✓ Installed {package}")

dependencies = [
    "nltk",
    "tensorflow",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib"
]

print("Installing required packages...")
for dep in dependencies:
    install_package(dep)

# Download NLTK data
print("\nDownloading NLTK resources...")
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

print("\nAll dependencies installed successfully!")