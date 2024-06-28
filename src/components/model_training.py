import os
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from dataclasses import dataclass
from src.utils import save_file_as_pickle
from src.utils import evaluate_model


@dataclass
class ModelTrainingConfig:
    train_model_pickle_file_path = os.path.join("artifacts/pickle", "model.pkl")


class ModelTraining:
    def __init__(self):
        self.model_train_config = ModelTrainingConfig()

    def initiate_model_training(self,transformed_train_dataset,transformed_test_dataset):
        try:
            models = {
                "LinearRegression": LinearRegression(),
                "StochasticLinearRegression": SGDRegressor(),
                "SupportVectorMachineRegression": SVR(),
                "KNearestNeighbourRegression": KNeighborsRegressor(),
                "DecisionTreeRegression": DecisionTreeRegressor(),
                "RandomForestRegression": RandomForestRegressor(),
                "AdaboostRegression": AdaBoostRegressor(),
                "GradientBoostRegression": GradientBoostingRegressor()
            }

            x_train,y_train,x_test,y_test=(
                transformed_train_dataset[:,:-1],
                transformed_train_dataset[:,-1],
                transformed_test_dataset[:,:-1],
                transformed_test_dataset[:,-1]
            )

            model_report = {}
            for model_name,model in models.items():
                model.fit(x_train, y_train)
                predicted_value = model.predict(x_test)
                r2 = evaluate_model(y_test, predicted_value)
                model_report[model_name] = r2

            best_model_name = max(model_report, key=model_report.get)
            best_model_r2=max(model_report.values())
            best_model = models[best_model_name]
            print("best model is:",best_model_name)
            print("r2 score for model is:" ,best_model_r2*100)
            save_file_as_pickle(best_model,self.model_train_config.train_model_pickle_file_path)
            return best_model
        except Exception as e:
            print(e)