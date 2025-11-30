import sys
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import sklearn
    print("✅ scikit-learn is installed")
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn import error: {e}")

try:
    import flask
    print("✅ Flask is installed")
    print(f"Flask version: {flask.__version__}")
except ImportError as e:
    print(f"❌ Flask import error: {e}")

try:
    import pandas as pd
    print("✅ pandas is installed")
except ImportError as e:
    print(f"❌ pandas import error: {e}")

try:
    import numpy as np
    print("✅ numpy is installed")
except ImportError as e:
    print(f"❌ numpy import error: {e}")