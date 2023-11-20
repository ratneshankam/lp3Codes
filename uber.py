# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_csv('uber.csv')

# df.head()

# df.isna().sum()

# df.fillna(0, inplace=True)

# df.isna().sum()

# plt.figure(figsize=(10, 5))
# sns.boxplot(data=df[['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']])
# plt.show()

# X = df.drop(columns=['key', 'pickup_datetime', 'passenger_count'])
# y = df['passenger_count']

# X
# y
# corr=df.corr()
# corr

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from sklearn.linear_model import LinearRegression
# linear_reg_model = LinearRegression()
# linear_reg_model.fit(X_train, y_train)
# y_pred_linear = linear_reg_model.predict(X_test)

# from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor(max_depth=2, random_state=0)
# regr.fit(X_train, y_train)
# y_pred_rf = regr.predict(X_test)

# from sklearn.metrics import r2_score, mean_squared_error

# r2_linear = r2_score(y_test, y_pred_linear)
# rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

# r2_rf = r2_score(y_test, y_pred_rf)
# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# print("Linear Regression - R-squared:", r2_linear)
# print("Linear Regression - RMSE:", rmse_linear)
# print()
# print("Random Forest Regression - R-squared:", r2_rf)
# print("Random Forest Regression - RMSE:", rmse_rf)

#Assignment5(Uber)

# Importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv("C:/Users/Ratnesh/Downloads/uber.csv")

df.head()  # Displaying the first few rows of the dataset

df.info()  # Displaying information about the dataset

df.columns  # Displaying the column names of the dataset

# Dropping unnecessary columns
df = df.drop(['Unnamed: 0', 'key'], axis=1)

df.head()

df.shape  # Displaying the total number of rows and columns

df.dtypes  # Displaying the data types of each column

df.info()

df.describe()  # Displaying statistics of each column

df.isnull().sum()

# Handling missing values
df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(), inplace=True)
df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(), inplace=True)

df.isnull().sum()

df.dtypes

# Converting pickup_datetime to datetime format and creating new time-related features
df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors='coerce')

df.dtypes

df = df.assign(hour=df.pickup_datetime.dt.hour,
               day=df.pickup_datetime.dt.day,
               month=df.pickup_datetime.dt.month,
               year=df.pickup_datetime.dt.year,
               dayofweek=df.pickup_datetime.dt.dayofweek)

df.head()

# Dropping the pickup_datetime column
df = df.drop('pickup_datetime', axis=1)

df.head()

df.dtypes

# Visualizing data distribution using boxplots
df.plot(kind="box", subplots=True, layout=(7, 2), figsize=(15, 20))
plt.show()

# Removing outliers using the Interquartile Range (IQR) method
def remove_outlier(df1, col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    df[col] = np.clip(df1[col], lower_whisker, upper_whisker)
    return df1
def treat_outliers_all(df1, col_list):
    for c in col_list:
        df1 = remove_outlier(df, c)
    return df1

df = treat_outliers_all(df, df.iloc[:, 0::])

df.plot(kind="box", subplots=True, layout=(7, 2), figsize=(15, 20))
plt.show()

# Calculating distance using Haversine formula
import haversine as hs
travel_dist = []
for pos in range(len(df['pickup_longitude'])):
    long1, lati1, long2, lati2 = [df['pickup_longitude'][pos], df['pickup_latitude'][pos],
                                  df['dropoff_longitude'][pos], df['dropoff_latitude'][pos]]
    loc1 = (lati1, long1)
    loc2 = (lati2, long2)
    c = hs.haversine(loc1, loc2)
    travel_dist.append(c)
print(travel_dist)
df['dist_travel_km'] = travel_dist
df.head()

# Filtering out distances greater than 130 km
df = df.loc[(df.dist_travel_km >= 1) | (df.dist_travel_km <= 130)]
print("Remaining observations in the dataset:", df.shape)

# Removing incorrect latitude and longitude values
incorrect_coordinates = df.loc[(df.pickup_latitude > 90) | (df.pickup_latitude < -90) |
                                (df.dropoff_latitude > 90) | (df.dropoff_latitude < -90) |
                                (df.pickup_longitude > 180) | (df.pickup_longitude < -180)|
                                (df.dropoff_longitude > 90) | (df.dropoff_longitude < -90)]

df.drop(incorrect_coordinates, inplace=True, errors='ignore')

df.head()

# Handling null values
df.isnull().sum()

sns.heatmap(df.isnull())

# Exploring correlation between features
corr = df.corr()

corr

fig, axis = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True)
plt.show()


# Selecting features and target variable
x = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'hour', 'day', 'month', 'year', 'dayofweek', 'dist_travel_km']]
y = df['fare_amount']

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Linear Regression model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()

regression.fit(X_train, y_train)

regression.intercept_

regression.coef_

prediction = regression.predict(X_test)

print(prediction)

y_test

# Model evaluation using R-squared
from sklearn.metrics import r2_score

r2_score(y_test, prediction)

# Model evaluation using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test, prediction)

MSE

RMSE = np.sqrt(MSE)

RMSE

# Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

y_pred

# Model evaluation for Random Forest Regressor
R2_Random = r2_score(y_test, y_pred)

R2_Random

MSE_Random = mean_squared_error(y_test, y_pred)

MSE_Random

RMSE_Random = np.sqrt(MSE_Random)

RMSE_Random
