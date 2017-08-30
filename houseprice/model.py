import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


data = pd.read_csv("train.csv",index_col="Id")
#data.info()
#data.head()


num_columns = ["MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF", "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]
txt_columns = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

print("{:d} txt columns plus {:d} num columns".format(len(txt_columns), len(num_columns)))

labels = data["SalePrice"]
data.drop("SalePrice", axis=1, inplace=True)
data.reset_index()
data = pd.get_dummies(data=data, dummy_na=True, columns=txt_columns)

txt_std_columns = list(data)
for col in num_columns:
    txt_std_columns.remove(col)

train_set, test_set, train_label, test_label = train_test_split(data, labels, random_state=42, train_size=0.8)


# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_columns)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(txt_std_columns)),
    ])
	
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


train_prepared = full_pipeline.fit_transform(train_set)

test_prepared = full_pipeline.fit_transform(test_set)

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty="l2",
                       learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(train_prepared, train_label)  # continues where it left off
    y_val_predict = sgd_reg.predict(test_prepared)
    val_error = mean_squared_error(y_val_predict, test_label)
    print(val_error)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

print(best_epoch)

test = pd.read_csv("test.csv",index_col="Id")

missed_cols = ['Utilities_NoSeWa', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'HouseStyle_2.5Fin', 'RoofMatl_ClyTile', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior2nd_Other', 'Heating_Floor', 'Heating_OthW', 'Electrical_Mix', 'GarageQual_Ex', 'PoolQC_Fa', 'MiscFeature_TenC']
for col in missed_cols:
    test[col] = [0] * 1459

test = pd.get_dummies(data=test, dummy_na=True, columns=txt_columns)




test_prepared = full_pipeline.fit_transform(test)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(train_prepared, train_label)


test['SalePrice'] = forest_reg.predict(test_prepared)
test['SalePrice'].to_csv("result.csv")