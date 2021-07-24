
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from fast_ml.model_development import train_valid_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
import dvc.api 

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

path ='data/AdSmartABdata.csv'
repo="C:/Users/Alt/workspace/Python/10academy/Week-2/abtest-mlops"
version="v5"


data_url= dvc.api.get_url(
        path=path,
        repo=repo,
        rev=version)
#mlflow.set_experiment('dvc')


def eval_metrics(actual, pred):
    mse=metrics.mean_squared_error(actual, pred)
    mae=metrics.mean_absolute_error(actual, pred)
    rmse=np.sqrt(metrics.mean_squared_error(actual, pred))
    variance=metrics.explained_variance_score(actual, pred)
    return mse, mae, rmse, variance

def model_select(model):
    if model == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        mod = LogisticRegression()

    elif model == 'DecisionTreeRegressor':
        from sklearn.tree import DecisionTreeRegressor
        mod = DecisionTreeRegressor()
        
    elif model == 'XGBRegressor':
        from xgboost import XGBRegressor
        mod = XGBRegressor()
        
    return mod, model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        data_url
    )
    try:
        data = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to load dataset", e
        )

#     def clean_dataset(df):
#         assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#         df.dropna(inplace=True)
#         indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#         return df[indices_to_keep].astype(np.float64)
    
    # necessary data 
    data=data[['experiment', 'hour', 'device_make', 'answer']]
    lb = LabelEncoder()
    data['device_make'] = lb.fit_transform(data['device_make'])
    data['experiment'] = lb.fit_transform(data['experiment'])
    # Split the data into training, validation and test sets..
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(data, target='answer', train_size=0.7, valid_size=0.2, test_size=0.1)

    # The predicted column is "quality" which is a scalar from [3, 9]
#     train_x = train.drop(["quality"], axis=1)
#     test_x = test.drop(["quality"], axis=1)
#     train_y = train[["quality"]]
#     test_y = test[["quality"]]


     # log artifacts columns 
    cols_x=pd.DataFrame(list(data[['experiment', 'hour', 'device_make', 'answer']].columns))
    cols_x.to_csv('features.csv', header=False, index=False)
    #mlflow.log_artifact('feature.csv')
    
    cols_y=pd.DataFrame(list(data[['answer']].columns))
    cols_y.to_csv('target.csv', header=False, index=False)
    #mlflow.log_artifact('target.csv')
    
    
    #learning parameters 
#     alpha = 0.8
#     l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    mlflow.end_run()
    with mlflow.start_run():
        model = model_select('XGBRegressor')
        name = model[1]
        model=model[0]
        model.fit(X_train, y_train)

        predicted_answer = model.predict(X_valid)

        mse, mae, rmse, variance = eval_metrics(y_valid, predicted_answer)

        print(f"model is :{name}")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"Variance Score: {variance}")

        mlflow.log_param("model", name)
        mlflow.log_param("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("Variance Score", variance)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(predicted_answer[0], "model", registered_model_name=name+" abtesting")
        else:
            mlflow.sklearn.log_model(predicted_answer[0], "model")
