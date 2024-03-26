# Linear Regression for number of files vs sleep time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error  
import numpy as np

number_of_files = np.array([12, 36, 108, 324, 972])
sleep_time = np.array([2.59, 3.79, 9.17, 25.73, 55.5])

# Cubic Regression
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
number_of_files_3 = poly_features_3.fit_transform(number_of_files.reshape(-1,1))
print("Expand the original data with 3 dim",number_of_files_3)

lin_model = LinearRegression()
lin_model.fit(number_of_files_3, sleep_time)

# Print the Coefficients of model
#print(f'Linear Regression Model : y = {lin_model.coef_[0]}x + {lin_model.intercept_}')
print(f'Cubic Regression Model : {lin_model.coef_[0]}x^3 + {lin_model.coef_[1]}x^2 + {lin_model.coef_[2]}x + {lin_model.intercept_}')
print(f'Predicted Sleep Time : {lin_model.predict(number_of_files_3)}')
# Linear Regression Model : 0.055x + 3.352
# Cubic Regression Model : 0.06x^3 + 5.32x^2 - 6.35x + 1.66

# Regression Score
#print(f'Regression Score : {lin_model.score(number_of_files.reshape(-1,1), sleep_time)}')
print(f'Regression Score : {lin_model.score(number_of_files_3, sleep_time)}')
print(f'Mean Squared Error : {mean_squared_error(sleep_time, lin_model.predict(number_of_files_3))}')
# Regression Score : 0.99
# Mean Squared Error : 0.017
