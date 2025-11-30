import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

# Check and install required packages
required_packages = ['scikit-learn', 'flask']

print("🔧 Installing required packages...")
for package in required_packages:
    try:
        if package == 'scikit-learn':
            import sklearn
            print("✅ scikit-learn is already installed")
        elif package == 'flask':
            import flask
            print("✅ Flask is already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        install_package(package)

print("\n🎯 All packages installed! Running the application...")

# Now run the main application
exec(open('complete_churn_app.py').read())


