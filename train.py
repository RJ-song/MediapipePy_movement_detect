import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


df = pd.read_csv('output.csv')
df.head()
df.tail()
df[df['class'=='up']]

X=df.drop('class',axis=1)
y=df['class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1234 )
