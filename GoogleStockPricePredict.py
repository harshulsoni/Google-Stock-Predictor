import pandas as pd
import quandl
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import math, datetime
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

df=quandl.get('WIKI/GOOGL')
df.to_pickle('WIKI_GOOGL.pickle')
#df=pd.read_pickle('WIKI_GOOGL.pickle')


df=df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
#print df.columns.values.tolist()
df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100;
df['PCT_change']=(df['Adj. Open']-df['Adj. Close'])/df['Adj. Close']*100;
df=df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#print df['Adj. Close']

forecast_col='Adj. Close'

df.fillna(-99999, inplace=True)


#predict for 10% of Data
#forecast_out=int(math.ceil(0.01*len(df)))

forecast_out=30

#print len(df), forecast_out

df['label']=df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)
#print df.head()
X_All=np.array(df.drop(['label'], 1))
X_All=preprocessing.scale(X_All)

X=X_All[:-forecast_out]
X_toPredict=X_All[-forecast_out:]
y=np.array(df['label'])[:-forecast_out]
y_toPredict=np.array(df['label'])[-forecast_out:]
print (len(X), len(y))
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

clf=LinearRegression() #use 1 Thread
#clf=LinearRegression(n_jobs=-1) #use max threads available
clf.fit(X_train, y_train)

test_accuracy=clf.score(X_test, y_test)
train_accuracy=clf.score(X_train, y_train)
#print test_accuracy, train_accuracy
#print X_test
#X_test_predicted=clf.predict(X_test)

Predicted_Value=clf.predict(X_toPredict)

print (Predicted_Value)


last_date_available=df.dropna().iloc[-1].name
last_unix_timestamp=last_date_available.timestamp()
one_day_sec=86400
predict_date=last_unix_timestamp

df.set_value(datetime.datetime.fromtimestamp(predict_date+one_day_sec), 'label', Predicted_Value[0])
#df.loc[datetime.datetime.fromtimestamp(predict_date+one_day_sec)]=[np.nan for _ in range(len(df.columns)-1)]+[Predicted_Value[0]]
print(df.loc[datetime.datetime.fromtimestamp(predict_date+one_day_sec)])
df['Forecast']=np.nan
for i in Predicted_Value:
    predict_date = predict_date + one_day_sec
    next_date=datetime.datetime.fromtimestamp(predict_date)
    df.set_value(next_date, 'Forecast', i)

#print (df.head())

df['Forecast'].plot()
df['label'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
