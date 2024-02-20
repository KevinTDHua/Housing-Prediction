
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

Xtr = pd.read_csv('Xtr_final.csv')
Xte = pd.read_csv('Xte_final.csv')
Ytr = pd.read_csv('Ytr.csv')

X_train, X_valid, Y_train, Y_valid = train_test_split(Xtr, Ytr["Sale_Amount"], test_size=0.2, random_state=42)
model = XGBRegressor(learning_rate=0.1, colsample_bytree=0.8)

model.fit(X_train, Y_train)

Y_train_pred = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
print("Training RMSE:", rmse_train)

Yte_pred = model.predict(Xte)
pred = pd.DataFrame(index=Xte.index)
pred["Sale_Amount"] = Yte_pred

pred_final = pred
pred_final["Sale_Amount"] = Yte_pred

pred_final.to_csv('pred_final.csv', index=False)


