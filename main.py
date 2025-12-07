import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

sales_df = pd.read_csv("advertising_and_sales_clean.csv")

y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()

# Create variable Z. A 2D array containing tv, radio and social media columns from sales_df.
Z = sales_df.drop("sales", axis=1).drop("influencer", axis=1).values

Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.3, random_state=42)

# Fit the model to the data
reg.fit(Z_train, y_train)

# Make predictions
y_pred = reg.predict(Z_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Compute R-squared
r_squared = reg.score(Z_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

# Create variable V. A 2D array containing radio and social media columns from sales_df.
V = sales_df.drop("sales", axis=1).drop("influencer", axis=1).drop("tv", axis=1).values

# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, V, y, cv=kf)

# Print scores
print("Cross validation scores: " + str(cv_scores))

# Print the mean
print("Mean of cross validation scores: " + str(np.mean(cv_scores)))

# Print the standard deviation
print("Standard deviation of cross validation scores: " + str(np.std(cv_scores)))

# Print the 95% confidence interval
print("95% CI of cross validation scores: " + str(np.quantile(cv_scores, [0.025, 0.975])))

# Assign values to alpha to use in the Ridge regression model
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
    # Create a Ridge regression model
    ridge = Ridge(alpha=alpha)

    # Fit the data
    ridge.fit(Z_train, y_train)

    # Obtain R-squared
    score = ridge.predict(Z_test)
    ridge_scores.append(ridge.score(Z_test, y_test))
print(ridge_scores)

sales_columns = ['tv', 'radio', 'social_media']

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(Z, y).coef_

# Compute and print the coefficients. Plot and show bar chart displaying lasso coefs for each feature.
lasso_coef = lasso.fit(Z, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
