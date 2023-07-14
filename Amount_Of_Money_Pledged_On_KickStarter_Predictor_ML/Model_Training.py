import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import explained_variance_score as evs
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor  
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

# Data Import
df = pd.read_csv('kickstarter_projects.csv')


# EDA
print(df.dtypes)

print(df.isnull().sum())


# Data Transformation
cols_to_drop = ['Name', 'Launched', 'Deadline', 'ID']
df = df.drop(cols_to_drop, axis=1)

le = LabelEncoder()

for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = le.fit_transform(df[c])
    df[c] = df[c].astype('float')
print(df)


# Train Test Split
features = df.drop('Pledged', axis=1)
target = df['Pledged']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.25, random_state= 42)


# Model Training
models = [DecisionTreeRegressor(), ExtraTreeRegressor(), LinearRegression(), PoissonRegressor(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]

for m in models:
    print(m)
    m.fit(X_train,Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy is : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {evs(Y_test, pred_test)}')