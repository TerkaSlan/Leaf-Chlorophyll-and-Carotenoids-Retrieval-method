import os
import random
from typing import Tuple

import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 2024
random.seed(RANDOM_STATE)

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="X does not have valid feature names"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of array concatenation with empty entries is deprecated",
)

from datetime import datetime
from typing import Tuple

import joblib
import pandas as pd
import rasterio
from tqdm import tqdm


def read_multiband_tif_as_array(tif_path):
    # Open the .tif file using rasterio
    with rasterio.open(tif_path) as dataset:
        # Read the entire dataset's data into a 3D NumPy array
        # Each band will be read into a separate layer in the array
        array = dataset.read()  # Read all bands

    return array


def S2_band_wavelength_mapping():
    return [496.54, 560.01, 664.45, 703.89, 740.22, 782.47, 864.8, 1613.7, 2202.4]


def load_valid_dataset(dir_path, vegetation_parameter):
    if vegetation_parameter == "Cab":
        vegetation_parameter = "Chl AB"
    df = pd.read_excel(f"{dir_path}/lanzhot_split.xlsx")
    param_min = df[vegetation_parameter].min()
    param_max = df[vegetation_parameter].max()
    params_train = df.query('Subset == "train"')[vegetation_parameter]
    params_test = df.query('Subset == "test"')[vegetation_parameter]

    hls_bands = ["B2", "B3", "B4", "B8A", "B11", "B12"]
    data_train = df.query('Subset == "train"')[hls_bands]
    data_test = df.query('Subset == "test"')[hls_bands]
    data_train = data_train[hls_bands]
    data_test = data_test[hls_bands]

    hls_wavelengths = [490, 560, 665, 865, 1610, 2190]
    hls = {k: v for k, v in zip(hls_bands, hls_wavelengths)}
    data_train = data_train.rename(columns=hls)
    data_test = data_test.rename(columns=hls)
    return data_train, data_test, params_train, params_test, param_min, param_max


def get_unique_filename(base_filename, extension):
    if "TIMESTAMP" not in os.environ:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = os.environ["TIMESTAMP"]

    counter = 0
    unique_filename = f"{base_filename}_{counter}_{timestamp}.{extension}"
    while os.path.exists(unique_filename):
        counter += 1
        unique_filename = f"{base_filename}_{counter}_{timestamp}.{extension}"
    return unique_filename


def transform_train(
    data_train, name, n_components=None, method="PCA", save=False, save_path=None
):
    if save:
        assert save_path is not None
    # Standardize the features
    transform_methods = []
    if method == None:
        data_train_pca = data_train
    if method != None:
        scaler = StandardScaler()
        data_train_pca = scaler.fit_transform(data_train)
        transform_methods.append(scaler)
        if save:
            scaler_filename = get_unique_filename(
                f"{save_path}/models/scaler_feat-{data_train.shape[1]}_{name}_{method}_{n_components}",
                "pkl",
            )
            joblib.dump(scaler, scaler_filename)

    # Apply PCA
    if method == "PCA":
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        data_train_pca = pca.fit_transform(data_train_pca)
        transform_methods.append(pca)
        if save:
            pca_filename = get_unique_filename(
                f"{save_path}/models/pca_feat-{data_train.shape[1]}_{name}_{method}_{n_components}",
                "pkl",
            )
            joblib.dump(pca, pca_filename)

    elif method == "SVD":
        tsvd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        transform_methods.append(tsvd)
        data_train_pca = tsvd.fit_transform(data_train_pca)
        if save:
            tsvd_filename = get_unique_filename(
                f"{save_path}/models/tsvd_feat-{data_train.shape[1]}_{name}_{method}_{n_components}",
                "pkl",
            )
            joblib.dump(tsvd, tsvd_filename)

    return data_train_pca, transform_methods


def transform(
    data_train,
    data_test,
    name,
    other_data=None,
    n_components=None,
    method="PCA",
    save=False,
    save_path=None,
):
    if save:
        assert save_path is not None
    # Standardize the features
    if method == None:
        data_train_pca = data_train
        data_test_pca = data_test
    if method != None:
        scaler = StandardScaler()
        data_train_pca = scaler.fit_transform(data_train)
        data_test_pca = scaler.transform(data_test)

        if save:
            scaler_filename = get_unique_filename(
                f"{save_path}/models/scaler_feat-{data_train.shape[1]}_{name}_{method}_{n_components}",
                "pkl",
            )
            joblib.dump(scaler, scaler_filename)

    if other_data is not None:
        if isinstance(other_data, (list)):
            other_data_prep = []
            for d in other_data:
                other_data_prep.append(scaler.transform(d))
        else:
            other_data = scaler.transform(other_data)

    if method == "PCA":
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        data_train_pca = pca.fit_transform(data_train_pca)
        data_test_pca = pca.transform(data_test_pca)
        if save:
            pca_filename = get_unique_filename(
                f"{save_path}/models/pca_feat-{data_train.shape[1]}_{name}_{method}_{n_components}",
                "pkl",
            )
            joblib.dump(pca, pca_filename)
        if other_data is not None:
            if isinstance(other_data, (list)):
                other_data_pca = []
                for d in other_data_prep:
                    other_data_pca.append(pca.transform(d))
            else:
                other_data = scaler.transform(other_data)
    elif method == "SVD":
        tsvd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        data_train_pca = tsvd.fit_transform(data_train_pca)
        data_test_pca = tsvd.transform(data_test_pca)
        if save:
            tsvd_filename = get_unique_filename(
                f"{save_path}/models/tsvd_feat-{data_train.shape[1]}__{name}_{method}_{n_components}",
                "pkl",
            )
            joblib.dump(tsvd, tsvd_filename)
        if other_data is not None:
            if isinstance(other_data, (list)):
                other_data_pca = []
                for d in other_data_prep:
                    other_data_pca.append(tsvd.transform(d))
            else:
                other_data = scaler.transform(other_data)

    if other_data is not None:
        return data_train_pca, data_test_pca, other_data_pca
    else:
        return data_train_pca, data_test_pca


def load_data_LUT(
    lut_data_path: str, vegetation_parameter: str, n_files=None
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame([])
    params = pd.Series([])
    loaded_files = []
    data_path = f"{lut_data_path}/LUT-HLS/"
    files = os.listdir(data_path)
    random.shuffle(files)
    for i, filename in tqdm(enumerate(files)):
        if filename.startswith("data-original"):
            df = pd.concat([df, pd.read_csv(f"{data_path}/{filename}", sep=" ")])
            loaded_files.append(filename)
            labels_filename = filename.replace("data-original", "labels")
            if os.path.exists(f"{data_path}/{labels_filename}"):
                params = pd.concat(
                    [
                        params,
                        pd.read_csv(f"{data_path}/{labels_filename}", sep=" ")[
                            vegetation_parameter
                        ],
                    ]
                )
        if n_files is not None and i >= n_files:
            break
    df = df.reset_index(drop=True)
    params = params.reset_index(drop=True)
    df.columns = df.columns.astype(int)
    return df, params


def train_single(
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    name,
    preprocessing_method,
    k,
    save=False,
    save_path=None,
):

    if save:
        assert save_path is not None
    if isinstance(X_train, pd.Series) or X_train.shape[1] == 1:
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)
    X_train, X_test = transform(
        X_train,
        X_test,
        name,
        n_components=k,
        method=preprocessing_method,
        save=save,
        save_path=save_path,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    return test_rmse, y_pred, model
