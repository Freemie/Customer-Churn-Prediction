import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=== SIMPLE TEST ===")

# Create simple synthetic data
np.random.seed(42)
n_samples = 1000

data = {
    'tenure': np.random.randint(1, 60, n_samples),
    'monthly_charges': np.random.uniform(30, 100, n_samples),
    'total_charges': np.random.uniform(100, 5000, n_samples),
    'contract': np.random.choice([0, 1, 2], n_samples),
    'support': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)

# Simple churn logic
df['churn'] = ((df['monthly_charges'] > 70) & (df['tenure'] < 24)).astype(int)

X = df[['tenure', 'monthly_charges', 'total_charges', 'contract', 'support']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Simple model accuracy: {accuracy:.4f}")
print("🎉 Basic functionality verified!")