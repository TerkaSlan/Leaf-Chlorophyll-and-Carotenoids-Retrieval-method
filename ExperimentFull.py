import csv
import logging

from Experiment import Experiment, mean_squared_error, os, pd, transform_train


class ExperimentFull(Experiment):

    def __init__(
        self,
        args,
        X_train,
        X_test,
        y_train,
        y_test,
        params_min,
        params_max,
        model,
        name,
    ):
        super().__init__(args, model=model, name=name)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.param_min = params_min
        self.param_max = params_max
        self.transform_methods = None
        self.test_run = args.test_run

    def run(self):
        assert "SAVEDIR" in os.environ
        subset = self.decide_subset(self.name, self.test_run)
        X = self.X_train.sample(frac=subset, random_state=self.random_state)
        y = self.y_train.loc[X.index]

        X, transform_methods = transform_train(
            X,
            self.name,
            n_components=self.k,
            method=self.preprocessing_method,
            save=True,
            save_path=os.environ["SAVEDIR"],
        )

        self.model.fit(X, y)
        self.transform_methods = transform_methods

    def test(self):
        assert (
            self.transform_methods is not None
        ), "self.transform_methods is none, run the model first"
        assert "SAVEDIR" in os.environ

        X_test = self.X_test.copy()
        for m in self.transform_methods:
            X_test = m.transform(X_test)

        y_pred = self.model.predict(X_test)
        test_rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        test_nrmse = test_rmse / (self.param_max - self.param_min)

        self.save_model(self.model, self.name, self.preprocessing_method, self.k, i=0)

        self.save_results(test_nrmse, y_pred)

        return test_nrmse, y_pred

    def decide_subset(self, model_name, test_run=False):
        fast_models = [
            "Dummy Regressor",
            "Linear Regression",
            "Ridge Regression",
            "Lasso",
            "SGDRegressor",
            "K-Nearest Neighbors",
            "Decision Tree",
            "Random Forest",
            "Extra Trees Regressor",
        ]
        slow_models = [
            "AdaBoost Regressor",
            "Gradient Boosting",
            "Kernel Ridge Regression",
            "Support Vector Machine",
        ]
        if model_name in fast_models:
            return 1.0 if not test_run else 0.001  # Use the entire dataset
        elif model_name in slow_models:
            return 0.1 if not test_run else 0.001  # Use 10% of the dataset
        else:
            raise ValueError("Model name not recognized")
