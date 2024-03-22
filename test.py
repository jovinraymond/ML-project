import mlflow
import joblib

def test():
    logged_model = 'runs:/00b28a7872fc44c99e6b22fc065f3bfc/classifier'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    model = joblib.load_model('engine_model.pkl')
    data = [700, 2.493592, 11.79093, 3.178981, 84.14416, 81.63219]
    pred = logged_model.predict(pd.DataFrame(data))
    return pred

print(test())