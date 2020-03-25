import numpy as np 
import pandas as pd
import datetime

confirmed_data = pd.read_csv('time_series_19-covid-Confirmed.csv')
#death_data = pd.read_csv('time_series_2019-ncov-Deaths.csv')
#recovered_data = pd.read_csv('time_series_2019-ncov-Recovered.csv')

confirmed = confirmed_data.iloc[:,4:] 
dates = confirmed.keys()

world_cases = []

for i in dates:
    confirmed_sum = confirmed[i].sum()
    
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)



days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)



days_in_future = 10
c = len(dates)
future_forcast = np.array([i+c for i in range(days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

from sklearn.model_selection import train_test_split
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False) 

from sklearn.svm import SVR
regressor = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)
regressor.fit(X_train_confirmed,y_train_confirmed)
y_pred = regressor.predict(future_forcast)
