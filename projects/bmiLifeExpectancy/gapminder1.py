# TODO: Add import statements
from pandas import read_csv
from sklearn.linear_model import LinearRegression
import numpy as np
# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = read_csv('bmi_and_life_expectancy.csv')
print(bmi_life_data['BMI'].values[:,np.newaxis])

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data['BMI'].values[:,np.newaxis],bmi_life_data['Life expectancy'].values[:,np.newaxis])

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)

#print(laos_life_exp)