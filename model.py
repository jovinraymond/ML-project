import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
import mlflow
import mlflow.sklearn
import joblib
import subprocess

MLFLOW_TRACKING_URI = "sqlite:///mlsqlite.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
subprocess.Popen(["mlflow", "ui", "--port", "8080", "--backend-store-uri", MLFLOW_TRACKING_URI])

features =['Engine rpm','Lub oil pressure','Fuel pressure','Coolant pressure','lub oil temp','Coolant temp']
target_variable = 'Engine Condition'


engine_health_dataset = pd.read_csv(r"C:\Users\PAVILION\Desktop\ML\ML-project\engine_data.csv")

experiment_name = "engine-condition"
mlflow.set_experiment(experiment_name=experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

with mlflow.start_run(experiment_id=experiment.experiment_id):
    X = engine_health_dataset.drop(columns= 'Engine Condition')
    Y = engine_health_dataset['Engine Condition']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
    print(X.shape,Y.shape,X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
    classifier = KNeighborsClassifier(p=2)
    classifier.fit(X_train,Y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test,y_pred)
    mse = mean_squared_error(Y_test, y_pred)

    r2 = r2_score(Y_test, y_pred)

    print(f'accuracy*100: {accuracy}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')


    mlflow.log_param("features", features)
    mlflow.log_param("target_variable", target_variable)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)


    mlflow.sklearn.log_model(classifier, "classifier")

    joblib.dump(classifier, 'engine_model.pkl')

    model_path = 'engine_model.pkl'

    mlflow.log_artifact(model_path)
mlflow.end_run()