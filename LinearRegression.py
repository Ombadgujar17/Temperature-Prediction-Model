import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
data = pd.read_csv("temperatures.csv")
X = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP','OCT','NOV','DEC']]
y = data['ANNUAL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Mean Squared Error (MSE):",mean_squared_error(y_test, y_pred))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red',linewidth=2, label='Ideal Fit')
plt.title('Actual vs Predicted Annual Temperatures')
plt.xlabel('Actual Annual Temperature')
plt.ylabel('Predicted Annual Temperature')
plt.legend()
plt.show()