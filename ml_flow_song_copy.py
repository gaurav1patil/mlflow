import os
import pickle
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from urllib.parse import urlparse
import mlflow
mlflow.set_tracking_uri("http://20.46.247.134:5000")
mlflow.set_experiment("Gaurav-Experiment-02")
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    f1score = f1_score(actual,pred)
    accuracy = accuracy_score(actual, pred)
    return f1score, accuracy


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    file_path = '/train.csv'
    try:
        data = pd.read_csv(file_path, index_col = 0)
    except Exception as e:
        logger.exception(
            "Unable to find csv file. Error: %s", e
        )

    # null value columns
    null_col_list = []
    for col in filter((lambda x : data[x].isnull().sum() > 0), data.isnull().sum().index):
        null_col_list.append(col)

    # fill null values with median
    for col in null_col_list:
        median = data[col].median()
        data[col].fillna(median, inplace = True)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "song_popularity"
    train_x = train.drop(["song_popularity"], axis=1)
    test_x = test.drop(["song_popularity"], axis=1)
    train_y = train[["song_popularity"]]
    test_y = test[["song_popularity"]]

    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else None

    with mlflow.start_run():
        model = RandomForestClassifier(max_depth =max_depth, random_state=42)
        model.fit(train_x, train_y)

        filename = 'saved_model/finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

        predicted_qualities = model.predict(test_x)

        (f1score, accuracy) = eval_metrics(test_y, predicted_qualities)

        print(f"Random Forest model :")
        print(f"F1 score: {f1score}")
        print(f"Accuracy: {accuracy}")
       
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_metric("f1_score",f1score)
        mlflow.log_metric("accuracy_score",accuracy)
        

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForest")
            mlflow.log_artifact('saved_model/finalized_model.sav')
            print(mlflow.get_artifact_uri())
            
        else:
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact('saved_model/finalized_model.sav')
            print(mlflow.get_artifact_uri())