import argparse
import ast
import logging
import os
import pickle
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff

from Experiment import get_features
from helpers import load_valid_dataset, read_multiband_tif_as_array

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

greens = [
    "#80AF81",
    "#508D4E",
    "#1A5319",
]

oranges = ["#E48F45", "#994D1C", "#6B240C"]


def save_predicted_image_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_predicted_image_data(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None


def load_visualization_data(base_tif_path):
    tif_paths = [
        "2019_S30_Lanzhot_DOY116_CC43.tif",
        "2021_S30_Lanzhot_DOY172_CC0.tif",
        "2019_S30_Lanzhot_DOY206_CC0.tif",
        "2020_S30_Lanzhot_DOY296_CC43.tif",
    ]
    np_data_all = []
    for path in tif_paths:
        array_3d = read_multiband_tif_as_array(f"{base_tif_path}/S2_chosen/{path}")
        shapes = tuple((array_3d.shape[1], array_3d.shape[2]))
        np_data = array_3d.reshape(array_3d.shape[0], -1).T  # / 10_000
        np_data_all.append(np_data)

    logging.info(f"Loaded {len(np_data_all)} images")

    mask = tiff.imread(f"{base_tif_path}/lanzhot_forest_mask.tif")
    logging.info(f"Loaded mask with shape {mask.shape}")
    return np_data_all, shapes, mask


def feature_select_visualization_data(np_data_all, best_algo, param):
    n_features = parse_n_features(best_algo)
    features = get_features(param, int(n_features))

    np_data_all_feature_selected = []
    for np_data in np_data_all:
        np_data = pd.DataFrame(np_data, columns=[490, 560, 665, 865, 1610, 2190])
        np_data = np_data[features]
        np_data_all_feature_selected.append(np_data)
    return np_data_all_feature_selected


def parse_n_features(best_algo):
    pattern = r"feat-(\d+)"

    # Search for the pattern in the filename
    match = re.search(pattern, best_algo)

    # Extract the number if found
    result = None
    if match:
        result = match.group(1)
    return result


def get_best_model(df, base_path):
    base_path = os.path.join(base_path, "models")
    best_algo = df.sort_values(by=["nrmse"]).iloc[0].filename
    # Adjust the pattern to match the required filename structure
    parts = best_algo.split("_")

    if len(parts) > 5:
        pattern = f"{parts[3]}_{parts[4]}_{parts[5]}_{parts[6].split('.')[0]}"
    else:
        pattern = f"{parts[3]}_{parts[4].split('.')[0]}"
    logging.info(f"Looking for pattern: {pattern} in {base_path}")

    # List to hold matched filenames
    matched_files = []

    # Iterate over all files in the folder
    for filename in os.listdir(base_path):
        if pattern in filename and filename.endswith(".pkl"):
            matched_files.append(filename)

    logging.info(f"Found {len(matched_files)} matched files -- {matched_files}")

    scaler_files = [file for file in matched_files if "scaler" in file]
    transform_files = [
        file for file in matched_files if "pca" in file or "tsvd" in file
    ]
    model_files = [file for file in matched_files if "model" in file]

    def extract_number(file_name):
        return int(re.search(r"_(\d+)_", file_name).group(1))

    # Sort the lists by the extracted integer
    scaler_files_sorted = sorted(scaler_files, key=extract_number)
    transform_files_sorted = sorted(transform_files, key=extract_number)
    model_files_sorted = sorted(model_files, key=extract_number)

    def load_model(base_path, filename):
        # Load the model
        joblib_filename = f"{base_path}/{filename}"
        model = joblib.load(joblib_filename)
        return model

    models = []
    for m in model_files_sorted:
        models.append(load_model(base_path, m))

    scalers = []
    for m in scaler_files_sorted:
        scalers.append(load_model(base_path, m))

    transformers = []
    for m in transform_files_sorted:
        transformers.append(load_model(base_path, m))

    logging.info(
        f"Loaded {len(models)} models and {len(scalers)} scalers and {len(transformers)} transformers"
    )

    return models, scalers, transformers, best_algo


def predict_plot_heatmap_scaler(
    results1,
    results_dir1,
    results2,
    results_dir2,
    results3,
    results_dir3,
    np_data_orig,
    mask,
    shapes,
    param,
    data_valid_path,
    save_dir,
    cmap,
):
    image_names = ["26 Apr 2019", "21 Jun 2021", "18 Jul 2019", "22 Oct 2020"]

    fig, axs = plt.subplots(3, 4, figsize=(12, 7))
    fig.suptitle(
        f"Predicted {param} at Lanžhot for 2019-2021", fontsize=16, y=0.95
    )  # Add title to the figure, move it closer to plots
    _, _, valid_y, _, _, _ = load_valid_dataset(data_valid_path, param)

    row_titles = ["Statistical", "ALSS", "LUT"]  # Placeholders for row titles

    for j, (results, results_dir) in enumerate(
        zip([results1, results2, results3], [results_dir1, results_dir2, results_dir3])
    ):
        best_algo = results.sort_values(by=["nrmse"]).iloc[0].filename
        predictions_file = os.path.join(
            results_dir,
            f"{best_algo.split('.csv')[0]}_{image_names[0]}_predictions.pkl",
        )
        """
        if not os.path.exists(
            os.path.join(
                results_dir,
                f"{best_algo.split('.csv')[0]}_{image_names[0]}_predictions.pkl",
            )
        ):
            logging.info(f"Did not find predictions, model will be needed")
            models, scalers, transformers, best_algo = get_best_model(
                results, results_dir
            )
        """
        np_data = feature_select_visualization_data(np_data_orig, best_algo, param)
        for i, (name, img) in enumerate(zip(image_names, np_data)):
            predictions_file = os.path.join(
                results_dir, f"{best_algo.split('.csv')[0]}_{name}_predictions.pkl"
            )
            print(predictions_file)
            # Check if predictions already exist, otherwise predict
            if os.path.exists(predictions_file):
                logging.info(f"Found predictions in {predictions_file}, loading")
                pred_data = load_predicted_image_data(predictions_file)
            else:
                preds_all = []
                if scalers is not None and len(scalers) > 0:
                    for scaler in scalers:
                        preds_all.append(scaler.transform(img))
                if transformers is not None and len(transformers) > 0:
                    assert len(transformers) == len(preds_all)
                    preds_all_new = []
                    for transformer, preds in zip(transformers, preds_all):
                        preds_all_new.append(transformer.transform(preds))
                    preds_all = preds_all_new

                for i_m, model in enumerate(models):
                    if len(preds_all) < len(models):
                        pred = model.predict(img)
                        pred = np.where(pred < 0, valid_y.min(), pred)
                        preds_all.append(pred)
                    else:
                        pred = model.predict(preds_all[i_m])
                        pred = np.where(pred < 0, valid_y.min(), pred)
                        preds_all[i_m] = pred

                preds = np.mean(np.array(preds_all), axis=0)
                # Apply the mask
                pred_data = preds.reshape(shapes[0], shapes[1])
                pred_data = pred_data * mask
                save_predicted_image_data(pred_data, predictions_file)
            ax = axs[j, i]
            sns.heatmap(
                pred_data,
                ax=ax,
                cmap=cmap,
                cbar=False,
                cbar_ax=None,
                xticklabels=False,
                yticklabels=False,
                fmt=".2f",
                annot_kws={"fontsize": 12},
            )
            if j == 0:
                ax.set_title(name, fontsize=14)  # Add title to each column

        axs[j, 0].set_ylabel(
            row_titles[j], fontsize=14, rotation=0, labelpad=80
        )  # Add title to each row

    cbar_ax = fig.add_axes(
        [0.86, 0.05, 0.02, 0.7]
    )  # Define color bar axis position [left, bottom, width, height]

    if param == "Car":
        min_y = 2.5
        max_y = 7.0
    else:
        min_y = 10
        max_y = 60
    norm = plt.Normalize(vmin=min_y, vmax=max_y)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # dummy empty array for the scalar mappable
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{param}", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(right=0.8, top=0.9)  # Adjust right and top boundary
    fig.tight_layout(
        rect=[0, 0, 0.85, 0.9]
    )  # Ensure the layout fits well within the specified boundaries

    plt.savefig(f"{save_dir}/heatmap_{param}.png", dpi=300)
    logging.info(f"Saved heatmap to {save_dir}/heatmap_{param}.png")


def parse_numbers(numbers_str):
    try:
        return np.array(ast.literal_eval(numbers_str))
    except (ValueError, SyntaxError):
        return None


def plot_pairplot(
    df_valid_Cab,
    df_valid_Car,
    df_subset_Cab,
    df_subset_Car,
    df_full_Cab,
    df_full_Car,
    data_valid_path,
    save_dir,
):
    _, _, _, actual1, _, _ = load_valid_dataset(data_valid_path, "Cab")
    # Prepare data for subplot 2 (right subplot - "Car")
    _, _, _, actual2, _, _ = load_valid_dataset(data_valid_path, "Car")

    # Extract best predictions from the dataframes
    best_algo_preds_valid_cab = parse_numbers(
        df_valid_Cab.sort_values(by=["nrmse"]).iloc[0].y_preds
    )
    best_algo_preds_valid_car = parse_numbers(
        df_valid_Car.sort_values(by=["nrmse"]).iloc[0].y_preds
    )
    best_algo_preds_subset_cab = parse_numbers(
        df_subset_Cab.sort_values(by=["nrmse"]).iloc[0].y_preds
    )
    best_algo_preds_subset_car = parse_numbers(
        df_subset_Car.sort_values(by=["nrmse"]).iloc[0].y_preds
    )
    best_algo_preds_full_cab = parse_numbers(
        df_full_Cab.sort_values(by=["nrmse"]).iloc[0].y_preds
    )
    best_algo_preds_full_car = parse_numbers(
        df_full_Car.sort_values(by=["nrmse"]).iloc[0].y_preds
    )

    nrmse_valid_cab = round(
        df_valid_Cab.sort_values(by=["nrmse"]).iloc[0].nrmse * 100, 2
    )
    nrmse_valid_car = round(
        df_valid_Car.sort_values(by=["nrmse"]).iloc[0].nrmse * 100, 2
    )
    nrmse_subset_cab = round(
        df_subset_Cab.sort_values(by=["nrmse"]).iloc[0].nrmse * 100, 2
    )
    nrmse_subset_car = round(
        df_subset_Car.sort_values(by=["nrmse"]).iloc[0].nrmse * 100, 2
    )
    nrmse_full_cab = round(df_full_Cab.sort_values(by=["nrmse"]).iloc[0].nrmse * 100, 2)
    nrmse_full_car = round(df_full_Car.sort_values(by=["nrmse"]).iloc[0].nrmse * 100, 2)

    # Sample data
    x1 = actual1
    x2 = actual2

    # Create a figure and primary axis
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    # Plot data on primary x-axis and y-axis
    for i, (axis, algo_cab, algo_car, nrmse_cab, nrmse_car, method) in enumerate(
        zip(
            [axes[0], axes[1], axes[2]],
            [
                best_algo_preds_valid_cab,
                best_algo_preds_subset_cab,
                best_algo_preds_full_cab,
            ],
            [
                best_algo_preds_valid_car,
                best_algo_preds_subset_car,
                best_algo_preds_full_car,
            ],
            [nrmse_valid_cab, nrmse_subset_cab, nrmse_full_cab],
            [nrmse_valid_car, nrmse_subset_car, nrmse_full_car],
            ["Statistical", "ALSS", "LUT"],
        )
    ):
        y1 = algo_cab
        y2 = algo_car
        min_val = min(
            actual1.min(),
            algo_cab.min(),
            actual2.min(),
            algo_car.min(),
        )
        max_val = max(actual1.max(), algo_car.max())

        axis.scatter(x1, y1, label=f"Cab, nRMSE={nrmse_cab}", color=greens[i])
        axis.set_xlabel("X1-axis", color=greens[i])
        axis.set_ylabel("Y1-axis", color=greens[i])
        axis.tick_params(axis="x", labelcolor=greens[i])
        axis.tick_params(axis="y", labelcolor=greens[i])
        axis.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="gray",
            linestyle="--",
            linewidth=2,
        )
        axis.grid(True)
        axis.set_xticks(np.arange(0, max_val + 10, 10))

        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.legend(loc=(0, 0.1))

        max_val = max(x2.max(), y2.max())
        # Create a secondary y-axis

        ax2 = axis.twinx()
        ax2.scatter(x1, y2, color=oranges[i], label=f"Car, nRMSE={nrmse_car}")
        ax2.set_ylabel("Y2-axis", color=oranges[i])
        ax2.tick_params(axis="y", labelcolor=oranges[i])
        ax2.set_yticks(np.arange(0, max_val + 1, 1))
        ax2.legend(loc=(0, 0))
        ax2.set_ylabel("Predicted")

        # Create a secondary x-axis
        ax3 = axis.twiny()
        ax3.set_xlabel("X2-axis", color=oranges[i])
        ax3.tick_params(axis="x", labelcolor=oranges[i])
        ax3.set_xticks(np.arange(0, max_val + 1, 1))
        ax3.set_xlabel("Actual")
        axis.set_title(method)

    # Add a title for the plot
    plt.suptitle(
        "Actual vs. predicted values, algorithm and hyperparameters with the best performance"
    )
    fig.tight_layout()  # Adjust layout to make room for the secondary axes
    plt.savefig(f"{save_dir}/pairplot.png", dpi=300)
    logging.info(f"Saved pairplot to {save_dir}/pairplot.png")


def plot_boxplots(
    df_Cab, df_Car, df_subset_Cab, df_subset_Car, df_full_Cab, df_full_Car, save_dir
):
    def drop_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

    # Filter data for nrmse < 1
    filtered_df1 = df_subset_Cab.query("nrmse < 1")
    filtered_df2 = df_full_Cab.query("nrmse < 1")
    filtered_df3 = df_subset_Car.query("nrmse < 1")
    filtered_df4 = df_full_Car.query("nrmse < 1")
    filtered_df5 = df_Cab.query("nrmse < 1")
    filtered_df6 = df_Car.query("nrmse < 1")

    filtered_df1["nrmse"] *= 100
    filtered_df2["nrmse"] *= 100
    filtered_df3["nrmse"] *= 100
    filtered_df4["nrmse"] *= 100
    filtered_df5["nrmse"] *= 100
    filtered_df6["nrmse"] *= 100

    # Drop outliers
    filtered_df1 = drop_outliers(filtered_df1, "nrmse")
    filtered_df2 = drop_outliers(filtered_df2, "nrmse")
    filtered_df3 = drop_outliers(filtered_df3, "nrmse")
    filtered_df4 = drop_outliers(filtered_df4, "nrmse")
    filtered_df5 = drop_outliers(filtered_df5, "nrmse")
    filtered_df6 = drop_outliers(filtered_df6, "nrmse")

    # Add a column to distinguish between the DataFrames
    filtered_df1["Source"] = "Cab (ALSS)"
    filtered_df2["Source"] = "Cab (LUT)"
    filtered_df3["Source"] = "Car (ALSS)"
    filtered_df4["Source"] = "Car (LUT)"
    filtered_df5["Source"] = "Cab (Statistical)"
    filtered_df6["Source"] = "Car (Statistical)"

    # Concatenate the DataFrames
    combined_df = pd.concat(
        [
            filtered_df1,
            filtered_df2,
            filtered_df3,
            filtered_df4,
            filtered_df5,
            filtered_df6,
        ]
    )

    custom_colors = {
        "Cab (Statistical)": greens[0],
        "Cab (ALSS)": greens[1],
        "Cab (LUT)": greens[2],
        "Car (Statistical)": oranges[0],
        "Car (ALSS)": oranges[1],
        "Car (LUT)": oranges[2],
    }

    # Create the boxplot
    plt.figure(figsize=(10, 3))
    combined_df = combined_df.sort_values(by="feature")
    source_order = [
        "Cab (Statistical)",
        "Cab (ALSS)",
        "Cab (LUT)",
        "Car (Statistical)",
        "Car (ALSS)",
        "Car (LUT)",
    ]
    ax = sns.boxplot(
        x="feature",
        y="nrmse",
        hue="Source",
        data=combined_df,
        hue_order=source_order,
        palette=custom_colors,
    )

    # Set z-order to make boxplots appear above grid lines
    for artist in ax.artists:
        artist.set_zorder(3)  # Higher z-order for boxplots

    # Adjust the grid lines z-order
    ax.grid(
        True, linestyle="--", linewidth=0.5, zorder=0
    )  # Lower z-order for grid lines

    plt.legend(ncol=3)
    # Add title and labels
    plt.title("Effect of feature selection, all algorithms")
    plt.xlabel("Number of wavelengths used")
    plt.ylabel("nRMSE (%)")
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{save_dir}/boxplots.png", dpi=300)


def plot_barchart(
    df_valid_Cab,
    df_valid_Car,
    df_subset_Cab,
    df_subset_Car,
    df_full_Cab,
    df_full_Car,
    save_dir,
):

    def apply(df_):
        df_ = df_.copy()
        df_["nrmse"] *= 100
        df_ = df_.query("nrmse < 100")
        idx_df1 = df_.groupby(["base_algorithm"])["nrmse"].idxmin()
        df_ = df_.loc[idx_df1]
        df_ = df_[~df_["base_algorithm"].str.contains("Dummy")]
        df_ = df_[~df_["base_algorithm"].str.contains("Kernel")]
        return df_

    df1 = apply(df_valid_Cab)
    df2 = apply(df_valid_Car)
    df_subset1 = apply(df_subset_Cab)
    df_subset2 = apply(df_subset_Car)
    # return df1, df2, df_subset1

    # Create the figure and axes
    _, axs = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

    # Get unique algorithms present in both datasets
    algorithms = df1["base_algorithm"].unique()

    # Plot bars for Cab and Car
    bar_width = 0.2
    index = np.arange(len(algorithms))
    cab_means = [
        df1.query("base_algorithm == @algo")["nrmse"].mean() for algo in algorithms
    ]
    car_means = [
        df2.query("base_algorithm == @algo")["nrmse"].mean() for algo in algorithms
    ]
    cab_means_subset = [
        df_subset1.query("base_algorithm == @algo")["nrmse"].mean()
        for algo in algorithms
    ]
    car_means_subset = [
        df_subset2.query("base_algorithm == @algo")["nrmse"].mean()
        for algo in algorithms
    ]

    axs.axhline(
        y=df_full_Cab.sort_values(by="nrmse").iloc[0].nrmse * 100,
        color=greens[2],
        linestyle="--",
        linewidth=2,
        label="Cab (LUT, best perf.)",
        zorder=1,  # Line behind bars
    )
    axs.axhline(
        y=df_full_Car.sort_values(by="nrmse").iloc[0].nrmse * 100,
        color=oranges[2],
        linestyle="--",
        linewidth=2,
        label="Car (LUT, best perf.)",
        zorder=1,  # Line behind bars
    )

    bars1 = axs.bar(
        index,
        cab_means,
        bar_width,
        label="Cab (statistical)",
        color=greens[0],
        zorder=2,
    )
    bars3 = axs.bar(
        index + bar_width,
        cab_means_subset,
        bar_width,
        label="Cab (ALSS)",
        color=greens[1],
        zorder=2,
    )
    bars2 = axs.bar(
        index + bar_width * 2 + 0.05,
        car_means,
        bar_width,
        label="Car (statistical)",
        color=oranges[0],
        zorder=2,
    )
    bars4 = axs.bar(
        index + bar_width * 3 + 0.05,
        car_means_subset,
        bar_width,
        label="Car (ALSS)",
        color=oranges[1],
        zorder=2,
    )

    # Set labels, title, ticks, and legend
    axs.set_xlabel("Algorithm", fontsize=12)
    axs.set_ylabel("nRMSE (%)", fontsize=12)
    axs.set_title(
        "Performance of individual algorithms, all hyperparameters",
        fontsize=14,
    )
    axs.set_xticks(index + bar_width / 2 + 0.225)

    algorithms_display = [algo.replace("_Car_", "_") for algo in algorithms]
    algorithms_display = [algo.replace("Regressor", "") for algo in algorithms_display]
    algorithms_display = [algo.replace("Regression", "") for algo in algorithms_display]

    # Create a list to hold the final labels
    final_labels = []
    for i, algo in enumerate(algorithms_display):
        final_labels.append(algo.split("_")[0])
    axs.set_xticklabels(final_labels, rotation=45, ha="right")
    axs.legend(loc="lower right", ncol=3)
    axs.grid(True)

    axs.set_xticklabels(algorithms_display, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/barchart.png", dpi=300)
    logging.info(f"Saved bar chart to {save_dir}/barchart.png")


def get_results_df(folder_path):
    # List to hold dataframes
    dfs = []

    def extract_model_name(filename):
        # Assuming the model name is always the third part when splitting by '_'
        return filename.split("_")[4]

    # Iterate over all files in the folder
    for filename in os.listdir(os.path.join(folder_path, "results")):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, "results", filename)
            df = pd.read_csv(file_path)
            df["filename"] = filename  # Add filename as a new column
            df["base_algorithm"] = extract_model_name(filename)
            df["feature"] = (
                df["filename"]
                .str.split("_", expand=True)[3]
                .str.split("-", expand=True)[1]
            )
            dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    # Obsolete column name consistency and cleanup
    if "nrmse_ensemble" in combined_df.columns:
        combined_df["nrmse"] = combined_df["nrmse_ensemble"]
        combined_df.drop(columns=["nrmse_ensemble"], inplace=True)
    if "mean_y_preds" in combined_df.columns:
        combined_df["y_preds"] = combined_df["mean_y_preds"]
        combined_df.drop(columns=["mean_y_preds"], inplace=True)
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistical_Cab", type=str, required=True)
    parser.add_argument("--statistical_Car", type=str, required=True)

    parser.add_argument("--ALSS_Cab", type=str, required=True)
    parser.add_argument("--ALSS_Car", type=str, required=True)

    parser.add_argument("--LUT_Cab", type=str, required=True)
    parser.add_argument("--LUT_Car", type=str, required=True)

    parser.add_argument("--data_valid_path", default="data/valid", type=str)
    parser.add_argument(
        "--data_visualization_path", default="data/visualization", type=str
    )
    parser.add_argument("--savedir", required=True, type=str)
    args = parser.parse_args()

    save_dir = args.savedir
    os.makedirs(save_dir, exist_ok=True)

    results_dir_Cab = args.statistical_Cab
    results_dir_Car = args.statistical_Car

    df_statistical_Cab = get_results_df(args.statistical_Cab)
    df_statistical_Car = get_results_df(args.statistical_Car)
    df_ALSS_Cab = get_results_df(args.ALSS_Cab)
    df_ALSS_Car = get_results_df(args.ALSS_Car)
    df_LUT_Cab = get_results_df(args.LUT_Cab)
    df_LUT_Car = get_results_df(args.LUT_Car)

    plot_barchart(
        df_statistical_Cab,
        df_statistical_Car,
        df_ALSS_Cab,
        df_ALSS_Car,
        df_LUT_Cab,
        df_LUT_Car,
        save_dir=save_dir,
    )
    plot_boxplots(
        df_statistical_Cab,
        df_statistical_Car,
        df_ALSS_Cab,
        df_ALSS_Car,
        df_LUT_Cab,
        df_LUT_Car,
        save_dir=save_dir,
    )

    plot_pairplot(
        df_statistical_Cab,
        df_statistical_Car,
        df_ALSS_Cab,
        df_ALSS_Car,
        df_LUT_Cab,
        df_LUT_Car,
        data_valid_path=args.data_valid_path,
        save_dir=save_dir,
    )

    images, shapes, mask = load_visualization_data(args.data_visualization_path)

    predict_plot_heatmap_scaler(
        df_statistical_Cab,
        args.statistical_Cab,
        df_ALSS_Cab,
        args.ALSS_Cab,
        df_LUT_Cab,
        args.LUT_Cab,
        images,
        mask,
        shapes,
        param="Cab",
        data_valid_path=args.data_valid_path,
        save_dir=save_dir,
        cmap="Greens",
    )
    predict_plot_heatmap_scaler(
        df_statistical_Car,
        args.statistical_Car,
        df_ALSS_Car,
        args.ALSS_Car,
        df_LUT_Car,
        args.LUT_Car,
        images,
        mask,
        shapes,
        param="Car",
        data_valid_path=args.data_valid_path,
        save_dir=save_dir,
        cmap="Oranges",
    )
