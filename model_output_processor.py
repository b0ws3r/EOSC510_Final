import csv
import uuid
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from pandas import DataFrame
import datetime


def get_predictions_from_best_performer(input_data):
    best_performer = "/content/drive/MyDrive/Colab Notebooks/EOSC510_Final/2b1fe551-20b7-45c2-911e-338208fcdd15.p"
    loaded_models = pickle.load(open(best_performer, "rb"))
    all_tmins = []
    all_tmaxs = []
    for mod in loaded_models:
        pred_output = mod.predict(input_data)
        tmins = pred_output[:, 0]
        tmaxs = pred_output[:, 1]
        all_tmins.append(tmins)
        all_tmaxs.append(tmaxs)

    # get ensemble mean for each variable
    all_tmins = np.asarray(all_tmins)
    tmin_mean_predicted_output = all_tmins.mean(axis=0)
    all_tmaxs = np.asarray(all_tmins)
    tmax_mean_predicted_output = all_tmaxs.mean(axis=0)

    data = {"pred_tmin": tmin_mean_predicted_output, "pred_tmax": tmax_mean_predicted_output}
    predictions_df = pd.DataFrame(data)
    print(predictions_df.head(3))
    return predictions_df


def filter_on_date(df, year, month, day):
    df_filtered = df
    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        date_filter = df["Date"] == pd.Timestamp(year, month, day)
        df_filtered = df[date_filter]
    else:
        print("No date column available to filter on")
    return df_filtered


# we get output y_hats for 1 date. need to compare them to orig
def join_data_to_original(predictions, original_all_columns, date_filter=None, ):
    df_new = original_all_columns.copy()

    df_new["pred_tmin"] = predictions["pred_tmin"]
    df_new["pred_tmax"] = predictions["pred_tmax"]
    print(df_new.head(3))
    return df_new


# forecasted TMIN , Corrected TMIN, Actual TMIN
# gfs_day (april 15)
wdir = 'data/model_outputs'
x_val_dir = f'{wdir}/validation_all_columns.p'

joined_validation_predictions_dir = f"{wdir}/validation_data_and_predictions.p"

predictions_df = get_predictions_from_best_performer(x_val)
joined_data = join_data_to_original(predictions_df, val)



pickle.dump(joined_validation_predictions, open(joined_validation_predictions, "wb"))

# tmin_fcst
# tmin_actual

# tmax_fcst
# tmax_pred
# tmax_actual


# map of error over forecast area
# todo
joined_data["reconstruct_pred"] = joined_data["pred_tmin"] - joined_data["tmin_K"]
joined_data["point_error_tmin"] = joined_data["pred_tmax"] - joined_data["tmax_K"]
