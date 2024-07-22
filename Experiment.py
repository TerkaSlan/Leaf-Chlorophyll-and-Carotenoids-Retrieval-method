import os

import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from helpers import *

models = [
    ("Dummy Regressor", DummyRegressor()),
    ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5)),
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge(random_state=42)),
    ("Lasso", Lasso(random_state=42)),
    (
        "Support Vector Machine",
        SVR(
            kernel="linear",
            max_iter=1000,
            cache_size=5000,
            shrinking=True,
            tol=0.01,
            epsilon=0.1,
        ),
    ),
    ("SGDRegressor", SGDRegressor(random_state=42)),
    ("Decision Tree", DecisionTreeRegressor(random_state=42)),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("Extra Trees Regressor", ExtraTreesRegressor(random_state=42)),
    ("AdaBoost Regressor", AdaBoostRegressor(random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
]


def get_features(param, n_features, array=False):
    # Features based on RF feature importances on valid data
    features = {
        "Car": [560, 865, 490, 2190, 665, 1610],
        "Cab": [560, 665, 865, 2190, 490, 1610],
    }

    # Get the top n_features for the given param
    top_features = features[param][:n_features]

    if array:
        # Create a full list of all possible features
        all_features = sorted(
            features[param]
        )  # Assuming all possible features are listed here
        # Create a boolean array indicating presence of top features
        boolean_array = [feature in top_features for feature in all_features]
        return boolean_array

    return top_features


def concat_list_as_string(lst, separator="-"):
    # Convert each integer in the list to a string
    str_lst = map(str, lst)
    # Join the string representations with the specified separator
    result = separator.join(str_lst)
    return result


class Experiment:
    def __init__(self, args, model, name):
        self.random_state = args.random_state
        self.vegetation_parameter = args.param
        self.top_n = args.top_n
        self.k = args.k
        self.preprocessing_method = args.preprocessing_method
        self.data_dir = args.data_dir
        self.data_valid_dir = args.data_valid_dir
        self.test_run = args.test_run
        self.model = model
        self.name = name
        self.features = args.features
        self.is_finished = False

        self.savedir = os.environ["SAVEDIR"]
        # os.environ["SAVEDIR"] = self.savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        else:
            if self.is_experiment_finished():
                print(
                    f'Experiment with "feat-{len(self.features)}_{self.name}_{self.preprocessing_method}_{self.k}" is already finished, skipping.'
                )
                self.is_finished = True

        self.timestamp = args.timestamp
        os.environ["TIMESTAMP"] = self.timestamp
        self.top_nrmse_indices = []

    def save_model(self, model, name, preprocessing, k, i):
        # Save the model
        joblib_filename = f'{os.environ["SAVEDIR"]}/models/model_{i}_feat-{len(self.features)}_{name}_{preprocessing}_{k}_{self.timestamp}.pkl'
        joblib.dump(model, joblib_filename)

    def save_results(self, test_nrmse, y_pred):
        # Save nrmse_ensemble and mean_y_preds to DataFrame
        results_df = pd.DataFrame({"nrmse": [test_nrmse], "y_preds": [y_pred.tolist()]})

        # Apply the function to format the 'y_preds' column
        results_df.to_csv(
            f'{os.environ["SAVEDIR"]}/results/results_{self.timestamp}_feat-{len(self.features)}_{self.name}_{self.preprocessing_method}_{self.k}.csv',
            index=False,
        )

    def is_experiment_finished(self):
        if os.path.exists(f'{os.environ["SAVEDIR"]}/results'):
            results = os.listdir(f'{os.environ["SAVEDIR"]}/results')
            if any(
                [
                    f"feat-{len(self.features)}_{self.name}_{self.preprocessing_method}_{self.k}"
                    in r
                    for r in results
                ]
            ):
                return True
        return False
