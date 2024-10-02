import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page title
st.title("Linear Regression Example")

# Sidebar for input parameters
st.sidebar.header("Input Data")
num_points = st.sidebar.slider("Select number of data points", min_value=2, max_value=100, value=10)
x_min = st.sidebar.slider("Select minimum value for x", min_value=0, max_value=100, value=0)
x_max = st.sidebar.slider("Select maximum value for x", min_value=0, max_value=100, value=100)

# Linear regression parameters
a = st.sidebar.slider("Input slope (a)", min_value=-10.0, max_value=10.0, value=2.0)
b = st.sidebar.slider("Input intercept (b)", min_value=-10.0, max_value=10.0, value=0.0)
c = st.sidebar.slider("Input noise coefficient (c)", min_value=0.0, max_value=50.0, value=10.0)

# Generate random data
np.random.seed(0)
x_values = np.random.uniform(x_min, x_max, size=num_points)
noise = np.random.normal(0, c, size=num_points)
y_values = a * x_values + b + noise

# Display data points
st.write("Generated Data Points:")
data = pd.DataFrame({'X': x_values, 'Y': y_values})
st.write(data)

# Linear regression model
model = LinearRegression()
model.fit(data[['X']], data['Y'])

# Predictions
predicted_y = model.predict(data[['X']])
data['Predicted'] = predicted_y

# Plotting the results
st.subheader("Data Points and Linear Regression Model")
st.scatter_chart(data.set_index('X')[['Y']])  # Actual data points
st.line_chart(data.set_index('X')[['Predicted']])  # Linear regression predictions

# Display model parameters
st.subheader("Model Parameters")
st.write(f"Intercept: {model.intercept_}")
st.write(f"Slope: {model.coef_[0]}")
