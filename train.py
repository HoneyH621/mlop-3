
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# Load diabetes dataset
data = load_diabetes()

X = data.data
y = data.target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "diabetes_model.joblib")

print("Model trained and saved successfully!")
