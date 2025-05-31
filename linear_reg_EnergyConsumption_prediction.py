import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
df = pd.read_csv("Energy_consumption.csv")
df = df.drop(columns=['Timestamp','DayOfWeek'])
df['HVACUsage'] = df['HVACUsage'].str.lower().str.strip().map({'on': 1, 'off': 0})
df['LightingUsage'] = df['LightingUsage'].str.lower().str.strip().map({'on': 1, 'off': 0})

X = df[['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'HVACUsage', 'LightingUsage']]
y = df['EnergyConsumption']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#feature scaling(imp for ridge and lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Linear reg(multiple as more than 1 feature)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
#poly regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

pr =  LinearRegression()
pr.fit(X_train_poly,y_train)
y_pred_pr = pr.predict(X_test_poly)
#ridge reg(used to remove overfitting)
ridge=Ridge(alpha=1.0)
ridge.fit(X_train_scaled,y_train)
y_pred_ridge=ridge.predict(X_test_scaled)
#lasso reg (shrinks irrelevant features)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled,y_train)
y_pred_lasso=lasso.predict(X_test_scaled)

print("Model\t\t\tMSE\tR²")
print(f"Linear Regression\t{mean_squared_error(y_test, y_pred_lr):.2f}\t{r2_score(y_test, y_pred_lr):.2f}")
print(f"Polynomial Regression\t{mean_squared_error(y_test, y_pred_pr):.2f}\t{r2_score(y_test, y_pred_pr):.2f}")
print(f"Ridge Regression\t{mean_squared_error(y_test, y_pred_ridge):.2f}\t{r2_score(y_test, y_pred_ridge):.2f}")
print(f"Lasso Regression\t{mean_squared_error(y_test, y_pred_lasso):.2f}\t{r2_score(y_test, y_pred_lasso):.2f}")
