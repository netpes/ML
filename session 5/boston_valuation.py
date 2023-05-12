from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# Gather Data
boston_dataset = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

target = raw_df.values[1::2, 2]
data = pd.DataFrame(boston_dataset)
# data['PRICE'] = target
data = data.rename(
    columns={0: "CRIM", 1: "ZN", 2: "INDUS", 3: "CHAS", 4: "NOX", 5: "RM", 6: "AGE", 7: "DIS", 8: "RAD", 9: "TAX",
             10: "PTRATIO", 11: "B", 12: "LSTAT", 13: "MEDV"})
features = data.drop(['INDUS', 'AGE'], axis=1)
log_prices = np.log(target)
target = pd.DataFrame(log_prices, columns=['PRICE'])
property_stat = features.mean().values.reshape(1, 11)
regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_rooms, students_per_classroom, next_to_river=False, high_confidence=True):
    # configure property
    property_stat[0][4] = nr_rooms
    property_stat[0][10] = students_per_classroom
    if next_to_river:
        property_stat[0][3] = 1
    else:
        property_stat[0][3] = 0
    log_estimate = regr.predict(property_stat)

    # clac range
    if high_confidence:
        upper_bound = log_estimate + 2 * np.sqrt(MSE)
        lower_bound = log_estimate - 2 * np.sqrt(MSE)
        interval = 95
    else:
        upper_bound = log_estimate + np.sqrt(MSE)
        lower_bound = log_estimate - np.sqrt(MSE)
        interval = 68
        # Do |Y
    return log_estimate, upper_bound, lower_bound, interval


get_log_estimate(3, 16)


def get_dollar_estimate(rm, ptration, chas=False, large_rang=True):
    """

    :param rm: number of room
    :param ptration: number of students per teachers
    :param chas: True is the property is next to the river
    :param large_rang: True for 95% prediction False for 68%
    :return:
    """
    if rm < 1 or ptration:
        print("That is unrealistic try again")
        return
    Zillow_median = 583.3
    Scale_factor = Zillow_median / np.median(target)
    log_est, upper, lower, conf = get_log_estimate(rm, ptration, chas, large_rang)

    # convert to today dollars
    dollar_est = np.e ** log_est * 1000 * Scale_factor
    dollar_high = np.e ** upper * 1000 * Scale_factor
    dollar_low = np.e ** lower * 1000 * Scale_factor
    # round to dollar value
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_est, -3)
    low = np.around(dollar_est, -3)


get_dollar_estimate(2, 200, chas=True)
