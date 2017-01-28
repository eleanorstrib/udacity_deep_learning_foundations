import pandas
from sklearn import linear_model

bmi_life_data = pandas.read_csv('bmi_and_life_expectancy.csv')

x_bmi = bmi_life_data[['BMI']]
y_life_exp = bmi_life_data[['Life expectancy']]

bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(x_bmi, y_life_exp)

laos_life_exp = bmi_life_model.predict(21.07931)

print(laos_life_exp)
