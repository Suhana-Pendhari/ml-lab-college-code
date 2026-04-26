import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

data = {
    'Outlook':['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'],
    'Temperature':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Windy':['False','True','False','False','False','True','True','False','False','False','True','True','False','True'],
    'PlayGolf':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('PlayGolf', axis=1)
y = df['PlayGolf']

model = DecisionTreeClassifier()

param_grid = {
    'max_depth':[2,3,4,5],
    'criterion':['gini','entropy']
}

grid = GridSearchCV(model, param_grid, cv=5)

grid.fit(X,y)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
