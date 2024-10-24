import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import numpy as np

# Step 1: Load the CSV file
file_path = 'housing.csv'  # Replace with your CSV file path
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston = pd.read_csv(file_path, header=None, delimiter=r"\s+", names=column_names)

# Display the dataframe in Streamlit
st.title("Boston Housing Data")
st.write(boston)

# Extract features and target
X = boston['LSTAT'].values
Y = boston['MEDV'].values

# Scale the features and target
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X.reshape(-1, 1)).flatten()
y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1, 1)).flatten()

# Define the error function
def error(m, x, c, t):
    N = x.size
    e = sum(((m * x + c) - t) ** 2)
    return e * 1 / (2 * N)

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)

# Update function for gradient descent
def update(m, x, c, t, learning_rate):
    grad_m = sum(2 * ((m * x + c) - t) * x)
    grad_c = sum(2 * ((m * x + c) - t))
    m -= grad_m * learning_rate
    c -= grad_c * learning_rate
    return m, c

# Gradient descent function
def gradient_descent(init_m, init_c, x, t, learning_rate, iterations, error_threshold):
    m, c = init_m, init_c
    error_values = []
    for i in range(iterations):
        e = error(m, x, c, t)
        if e < error_threshold:
            st.write('Error less than the threshold. Stopping Gradient Descent.')
            break
        error_values.append(e)
        m, c = update(m, x, c, t, learning_rate)
    return m, c, error_values

# Parameters for gradient descent
init_m = 0.9
init_c = 0
learning_rate = 0.001
iterations = 250
error_threshold = 0.001

# Perform gradient descent
m, c, error_value = gradient_descent(init_m, init_c, xtrain, ytrain, learning_rate, iterations, error_threshold)

# Predictions
predicted = (m * xtest) + c
predicted = predicted.reshape(-1, 1)

# Inverse scaling
xtest_scaled = x_scaler.inverse_transform(xtest.reshape(-1, 1)).flatten()
ytest_scaled = y_scaler.inverse_transform(ytest.reshape(-1, 1)).flatten()
predicted_scaled = y_scaler.inverse_transform(predicted).flatten()

# Create DataFrame for display
results = pd.DataFrame({
    'X': xtest_scaled,
    'Target Y': ytest_scaled,
    'Predicted Y': predicted_scaled
})

# Display results
st.write("Predictions vs Actual Values")
st.write(results.round(decimals=2))

# Mean Squared Error
mse = mean_squared_error(ytest_scaled, predicted_scaled)
st.write(f"Mean Squared Error: {mse:.2f}")

# Creating a range of values for LSTAT for trend line
lstat_range = np.linspace(X.min(), X.max(), 100)
predicted_trend = (m * x_scaler.transform(lstat_range.reshape(-1, 1))) + c
predicted_trend = y_scaler.inverse_transform(predicted_trend.reshape(-1, 1)).flatten()

# Calculate standard deviation for the prediction intervals
std_dev = np.std(predicted_scaled - ytest_scaled)
lower_bound = predicted_trend - 1.96 * std_dev
upper_bound = predicted_trend + 1.96 * std_dev

# Creating the trend plot
trend_fig = go.Figure()

# Add actual values
trend_fig.add_trace(go.Scatter(x=boston['LSTAT'], y=boston['MEDV'], mode='markers', name='Actual Values', marker=dict(color='blue', opacity=0.6)))

# Add predicted trend
trend_fig.add_trace(go.Scatter(x=lstat_range, y=predicted_trend, mode='lines', name='Predicted Trend', line=dict(color='red')))

# Add confidence interval
trend_fig.add_trace(go.Scatter(x=lstat_range, y=lower_bound, mode='lines', name='Lower Bound', line=dict(color='lightgray', dash='dash')))
trend_fig.add_trace(go.Scatter(x=lstat_range, y=upper_bound, mode='lines', name='Upper Bound', line=dict(color='lightgray', dash='dash')))

# Update layout
trend_fig.update_layout(title='Predicted Trend Channel for LSTAT vs MEDV',
                        xaxis_title='LSTAT',
                        yaxis_title='MEDV')

# Display trend plot in Streamlit
st.plotly_chart(trend_fig)
