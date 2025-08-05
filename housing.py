import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

# ----------------------------------------
# Configuration
# ----------------------------------------

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# ----------------------------------------
# Data Fetching & Loading
# ----------------------------------------


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Download and extract the housing dataset."""
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    """Load housing data from the extracted CSV."""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ----------------------------------------
# Data Splitting
# ----------------------------------------


def stratified_split(data, test_size=0.2, random_state=42):
    """Split dataset into train and test sets using stratified sampling."""
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index].copy()
        strat_test_set = data.loc[test_index].copy()

    # Remove income_cat to return clean data
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


# ----------------------------------------
# EDA (Exploratory Data Analysis)
# ----------------------------------------


def explore_data(data):
    """Print dataset info and show histograms."""
    print("\nFirst few rows:\n", data.head())
    print("\nInfo:\n")
    print(data.info())
    print(
        "\nValue counts for 'ocean_proximity':\n",
        data["ocean_proximity"].value_counts(),
    )
    print("\nDescriptive stats:\n", data.describe())

    # Plot histograms
    data.hist(bins=50, figsize=(20, 15))
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# Main Execution
# ----------------------------------------

if __name__ == "__main__":
    fetch_housing_data()
    housing = load_housing_data()
    train_set, test_set = stratified_split(housing)
    housing = train_set.copy()
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=housing["population"] / 100,
        label="population",
        figsize=(10, 7),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
    )
    plt.legend()
    plt.show()
    corr_matrix = housing.select_dtypes(include=[np.number]).corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    attributes = [
        "median_house_value",
        "median_income",
        "total_rooms",
        "housing_median_age",
    ]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    housing.plot(
        kind="scatter",
        x="median_income",
        y="median_house_value",
        alpha=0.1,
    )
    plt.show()
    # print(f"\nTrain set size: {len(train_set)}")
    # print(f"Test set size: {len(test_set)}")

    explore_data(train_set)
