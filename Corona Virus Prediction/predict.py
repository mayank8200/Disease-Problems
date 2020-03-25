import numpy as np 
import pandas as pd
import datetime
import pickle


confirmed_data = pd.read_csv('time_series_19-covid-Confirmed.csv')
death_data = pd.read_csv('time_series_19-covid-Deaths.csv')
recovered_data = pd.read_csv('time_series_19-covid-Recovered.csv')

confirmed = confirmed_data.iloc[:,4:] 
death = death_data.iloc[:,4:]
recovered = recovered_data.iloc[:,4:]


dates = confirmed.keys()
world_cases = []
world_deaths = []
world_recovered = []

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = death[i].sum()
    recovered_sum = recovered[i].sum()
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    world_deaths.append(death_sum)
    world_recovered.append(recovered_sum)
    



days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
world_deaths = np.array(world_deaths).reshape(-1, 1)
world_recovered = np.array(world_recovered).reshape(-1, 1)




days_in_future = 10
c = len(dates)
future_forcast = np.array([i+c for i in range(days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

"""from sklearn.model_selection import train_test_split
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)""" 

from sklearn.svm import SVR
regressor = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)
regressor.fit(days_since_1_22,world_cases)
y_pred_cases = regressor.predict(future_forcast)
regressor1 = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)
regressor1.fit(days_since_1_22,world_deaths)
y_pred_deaths = regressor1.predict(future_forcast)
regressor2 = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)
regressor2.fit(days_since_1_22,world_recovered)
y_pred_recovered = regressor2.predict(future_forcast)

filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
filename = 'model1.pkl'
pickle.dump(regressor1, open(filename, 'wb'))
filename = 'model2.pkl'
pickle.dump(regressor2, open(filename, 'wb'))
