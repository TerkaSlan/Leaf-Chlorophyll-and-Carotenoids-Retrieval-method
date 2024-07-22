import logging

import numpy as np

from Experiment import Experiment, mean_squared_error, os, pd, tqdm, train_single


class ExperimentSubset(Experiment):

    def __init__(
        self,
        args,
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        params_min,
        params_max,
        model,
        name,
    ):
        super().__init__(args, model=model, name=name)
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.param_min = params_min
        self.param_max = params_max

    def run(self):
        if self.test_run:
            n_iters = 10
        else:
            n_iters = 10_000

        self.top_nrmse_indices = []
        np.random.seed(self.random_state)
        random_seeds = np.random.randint(0, 200_000_000, size=n_iters)

        for i in tqdm(range(n_iters)):
            X = self.X_train.sample(300, random_state=random_seeds[i])
            y = self.y_train.loc[X.index]

            rmse, _, _ = train_single(
                X,
                self.X_valid,
                y,
                self.y_valid,
                self.model,
                self.name,
                self.preprocessing_method,
                self.k,
            )
            nrmse = rmse / (self.param_max - self.param_min)

            if len(self.top_nrmse_indices) < self.top_n:
                self.top_nrmse_indices.append((nrmse, X.index))
                self.top_nrmse_indices.sort(key=lambda x: x[0])
            elif nrmse < self.top_nrmse_indices[-1][0]:
                if not np.any(
                    [np.all(X.index == i[1]) for i in self.top_nrmse_indices]
                ):
                    self.top_nrmse_indices[-1] = (nrmse, X.index)
                    self.top_nrmse_indices.sort(key=lambda x: x[0])

            for rank, (nrmse, indices) in enumerate(self.top_nrmse_indices, start=1):
                logging.info(f"Rank {rank}: NRMSE = {round(nrmse, 3)}")

        overlaps = []
        all_indices = set()
        for nrmse, indices in self.top_nrmse_indices:
            current_indices = set(indices)
            overlap = current_indices.intersection(all_indices)
            if overlap:
                overlaps.append(overlap)
            all_indices.update(current_indices)

        if overlaps:
            for overlap in overlaps:
                logging.info(
                    f"{self.name} {self.preprocessing_method} | Overlap: {list(overlap)}"
                )
        else:
            logging.info(
                f"{self.name} {self.preprocessing_method} |  No overlaps among the top 10 NRMSE indices."
            )

        for idx in self.top_nrmse_indices:
            logging.info(
                f"{self.name} {self.preprocessing_method} | top indices: {idx[1][:2]}"
            )

    def test(self):
        individual_test_nrmses = []
        individual_y_preds = []
        assert "SAVEDIR" in os.environ

        for i, indices in enumerate(self.top_nrmse_indices):
            nrmse, y_pred, model = train_single(
                self.X_train.loc[indices[1]],
                self.X_test,
                self.y_train.loc[indices[1]],
                self.y_test,
                self.model,
                self.name,
                self.preprocessing_method,
                self.k,
                save=True,
                save_path=os.environ["SAVEDIR"],
            )
            self.save_model(model, self.name, self.preprocessing_method, self.k, i)
            individual_test_nrmses.append(nrmse)
            individual_y_preds.append(y_pred)
            logging.info(
                f"{self.name} {self.preprocessing_method} | current y_pred: {y_pred[:10]}"
            )

        mean_y_preds = np.mean(np.array(individual_y_preds), axis=0)
        test_rmse = mean_squared_error(self.y_test, mean_y_preds, squared=False)
        nrmse_ensemble = test_rmse / (self.param_max - self.param_min)
        logging.info(
            f"{self.name} {self.preprocessing_method} | test nrmse (ensemble)={round(nrmse_ensemble, 3)}"
        )

        self.save_results(nrmse_ensemble, y_pred)
