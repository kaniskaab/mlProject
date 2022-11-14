import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
def prediction(temp,press,rain,wind):
    df=pd.read_csv('train.csv')
    df1=df.drop(['year','month','day','hour','wind_direction','Unnamed: 0'],axis='columns')
    df1=df1.iloc[1:,:]
    target=df1['PM2.5']
    le_temperature = LabelEncoder()
    le_pressure = LabelEncoder()
    le_rain = LabelEncoder()
    le_wind_speed = LabelEncoder()
    df1['Temperature_n'] = le_temperature.fit_transform(df1['temperature'])
    df1['Pressure_n'] = le_pressure.fit_transform(df1['pressure'])
    df1['Rain_n'] = le_rain.fit_transform(df1['rain'])
    df1['Wind_Speed_n'] = le_rain.fit_transform(df1['wind_speed'])
    inputs=df1.drop(['temperature','PM2.5','pressure','rain','wind_speed'],axis='columns')
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor()
    mean_value=target.mean()
    target.fillna(value=mean_value, inplace=True)
    target.replace([np.inf, -np.inf], np.nan, inplace=True)
    input1=inputs
    input2=target
    model.fit(input1,input2)
    model.score(input1,input2)*100
    # temp= -2.8
    # press=1018.3
    # rain=0
    # wind=1.4
    X1_test=np.array([temp,press,rain,wind])
    X1_test=X1_test.reshape((1,-1))
    return model.predict(X1_test)[0]
    # y_predicted=model.predict(input1)
    # y_predicted

    # mean_squared_score = mean_squared_error(input2,y_predicted)
    # print("MEAN SQUARED SCORE", mean_squared_score)
def ypredict():
    df=pd.read_csv('train.csv')
    df1=df.drop(['year','month','day','hour','wind_direction','Unnamed: 0'],axis='columns')
    df1=df1.iloc[1:,:]
    target=df1['PM2.5']
    le_temperature = LabelEncoder()
    le_pressure = LabelEncoder()
    le_rain = LabelEncoder()
    le_wind_speed = LabelEncoder()
    df1['Temperature_n'] = le_temperature.fit_transform(df1['temperature'])
    df1['Pressure_n'] = le_pressure.fit_transform(df1['pressure'])
    df1['Rain_n'] = le_rain.fit_transform(df1['rain'])
    df1['Wind_Speed_n'] = le_rain.fit_transform(df1['wind_speed'])
    inputs=df1.drop(['temperature','PM2.5','pressure','rain','wind_speed'],axis='columns')
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor

    model=RandomForestRegressor()
    mean_value=target.mean()
    target.fillna(value=mean_value, inplace=True)
    target.replace([np.inf, -np.inf], np.nan, inplace=True)
    input1=inputs
    input2=target
    model.fit(input1,input2)
    model.score(input1,input2)*100
    y_predicted=model.predict(input1)
    return y_predicted
def mss():
    df=pd.read_csv('train.csv')
    df1=df.drop(['year','month','day','hour','wind_direction','Unnamed: 0'],axis='columns')
    df1=df1.iloc[1:,:]
    target=df1['PM2.5']
    le_temperature = LabelEncoder()
    le_pressure = LabelEncoder()
    le_rain = LabelEncoder()
    le_wind_speed = LabelEncoder()
    df1['Temperature_n'] = le_temperature.fit_transform(df1['temperature'])
    df1['Pressure_n'] = le_pressure.fit_transform(df1['pressure'])
    df1['Rain_n'] = le_rain.fit_transform(df1['rain'])
    df1['Wind_Speed_n'] = le_rain.fit_transform(df1['wind_speed'])
    inputs=df1.drop(['temperature','PM2.5','pressure','rain','wind_speed'],axis='columns')
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor

    model=RandomForestRegressor()
    mean_value=target.mean()
    target.fillna(value=mean_value, inplace=True)
    target.replace([np.inf, -np.inf], np.nan, inplace=True)
    input1=inputs
    input2=target
    model.fit(input1,input2)
    model.score(input1,input2)*100
    y_predicted=model.predict(input1)
    mean_squared_score = mean_squared_error(input2,y_predicted)
    return mean_squared_score