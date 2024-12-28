import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace the path with your actual file path)
# Assuming the data is in a CSV file, e.g., 'real_estate.csv'
df = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\Realestate.csv")


# Display the first few rows of the data to verify it
print(df.head())

# Features (X) and target variable (y)
X = df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 
        'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but helps with convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model (e.g., using Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predicting a new value (example)
# Replace this with actual new data you want to predict the house price for
new_data = [[2014, 20, 400, 8, 24.985, 121.542]]  # Example new data
new_data_scaled = scaler.transform(new_data)
predicted_price = model.predict(new_data_scaled)
print(f'Predicted House Price: {predicted_price[0]}')
