import functools
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from model.covn_transformer_spectro_01 import config, get_model
from model.train import train

from model import generate_dataset

data_root = Path('../assets/emg_20221112')
info_file = pd.read_csv(data_root / 'assets-info.csv')
data_path = data_root / 'cropped'
checkpoint_path = f"../checkpoints/{config['model_name']}-all"

def custom_clean(X_data, y_data):
    extreme = abs(X_data) > 500
    X_data[extreme] = X_data.mean()
    mean, std = X_data.mean(), X_data.std()
    X_data = (X_data - mean) / std
    return X_data, y_data


model = get_model()

ckpt = tf.train.Checkpoint(model=model, )

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(ckpt_manager.latest_checkpoint)

model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.98,
                                                 epsilon=1e-9),
              metrics=['MSE', 'MAE'])

batch_size = 32
train_split, test_split, val_split = 0.7, 0.1, 0.2
size_val_2 = 4600
size_val_3 = 4400
# %%
train_labels = []
train_datas = []
for i in new_subject:
    print(i)
    row = info_file.iloc[i]

    name = row['name']
    date = datetime.strptime(row['time'], "%Y/%m/%d %H:%M").strftime('%Y-%m-%d')

    # curl_file_name = f'{name}_{date}_curl_calibrated.csv'
    curl_file_name = f'{name}_{date}_curl.csv'
    # emg_file_name = f'{name}_{date}_emg.csv'
    emg_file_name = f'{name}_{date}_emg_noabs.csv'

    curl_file = data_path / curl_file_name
    emg_file = data_path / emg_file_name

    datasets = generate_dataset(emg_file, curl_file, custom_split, custom_clean,
                                preprocessing="discretization", exclude_nan=False)

    train_dataset = datasets[0].unbatch().batch(1)

    ds_start, ds_end = [], []
    curl_data = np.genfromtxt(curl_file, delimiter=',')
    for idx in range(int(len(curl_data) / 50 / 3)):
        ds_start.append(idx * 50 * 3 + 75)
        ds_end.append(idx * 50 * 3 + 150)
    dss = []
    for start, end in zip(ds_start, ds_end):
        dss.append(train_dataset.skip(start).take(end - start))
    train_dataset = functools.reduce(lambda a, b: a.concatenate(b), dss).unbatch().batch(batch_size)

    train_label = np.concatenate([y for x, y in train_dataset], axis=0)
    train_data = np.concatenate([x for x, y in train_dataset], axis=0)

    train_labels.append(train_label)
    train_datas.append(train_data)

    fig, axs = plt.subplots(1)
    axs.plot(train_label)
    fig.set_size_inches(15 / 5400 * 27000, 15)
    plt.show()
# %%
for i in new_subject:
    print(i)
    row = info_file.iloc[i]

    name = row['name']
    date = datetime.strptime(row['time'], "%Y/%m/%d %H:%M").strftime('%Y-%m-%d')

    # curl_file_name = f'{name}_{date}_curl_calibrated.csv'
    curl_file_name = f'{name}_{date}_curl.csv'
    # emg_file_name = f'{name}_{date}_emg.csv'
    emg_file_name = f'{name}_{date}_emg_noabs.csv'

    curl_file = data_path / curl_file_name
    emg_file = data_path / emg_file_name

    datasets = generate_dataset(emg_file, curl_file, custom_split, custom_clean,
                                preprocessing="discretization", exclude_nan=False)

    train_dataset = datasets[0].unbatch().batch(1)

    val1_dataset = datasets[1].unbatch().batch(batch_size)

    val1_label = np.concatenate([y for x, y in val1_dataset], axis=0)

    model.evaluate(val1_dataset, use_multiprocessing=True, workers=12)
    pre_results1 = model.predict(val1_dataset, use_multiprocessing=True, workers=12)
    log_dir = f"../logs/fit/{config['model_name']}/{datetime.now().strftime('%Y%m%d-%H%M')}-{i}"

    ds_start, ds_end = [], []
    curl_data = np.genfromtxt(curl_file, delimiter=',')
    for idx in range(int(len(curl_data) / 50 / 3)):
        ds_start.append(idx * 50 * 3 + 75)
        ds_end.append(idx * 50 * 3 + 150)
    dss = []
    for start, end in zip(ds_start, ds_end):
        dss.append(train_dataset.skip(start).take(end - start))
    train_dataset = functools.reduce(lambda a, b: a.concatenate(b), dss).unbatch().batch(batch_size)

    for j in range(3):
        train(model, train_dataset, val1_dataset, epochs=1, lr=1e-5, log_dir=log_dir,
              loss=tf.keras.losses.MeanSquaredError())
        post_results1 = model.predict(val1_dataset, use_multiprocessing=True, workers=12)

        fig, axs = plt.subplots(3)
        fig.suptitle(f'Prediction on validation assets 1 for test {i}')
        fig.set_size_inches(15 / 5400 * pre_results1.shape[0], 15)
        axs[0].set_ylim(-1, 17)
        axs[0].plot(val1_label.reshape((-1, 5)), linewidth=0.6)
        axs[1].set_ylim(-1, 17)
        axs[1].plot(pre_results1, linewidth=0.6)
        axs[2].set_ylim(-1, 17)
        axs[2].plot(post_results1, linewidth=0.6)
        plt.savefig(f'../dist/figs-{config["model_name"]}/calibrated-{i}-{j}-re-val1.png')
        plt.show()

# %%
for i in [8]:
    row = info_file.iloc[i]

    name = row['name']
    date = datetime.strptime(row['time'], "%Y/%m/%d %H:%M").strftime('%Y-%m-%d')

    curl_file_name = f'{name}_{date}_curl.csv'
    emg_file_name = f'{name}_{date}_emg_noabs.csv'

    curl_file = data_path / curl_file_name
    emg_file = data_path / emg_file_name

    # curl_data = np.genfromtxt(curl_file, delimiter=',')[:-9000]
    curl_data = np.genfromtxt(curl_file, delimiter=',')[:-32000]

    fig, axs = plt.subplots(1)
    axs.plot(curl_data)
    fig.set_size_inches(30 * len(curl_data) / 5400, 15)
    for i in range(int(len(curl_data) / 50 / 6)):
        axs.axvline(x=i * 50 * 6 - 150, color='b')
        axs.axvline(x=i * 50 * 6 + 75, color='g')
        axs.axvline(x=i * 50 * 6, color='r')
    plt.show()
# %%
for i in range(30):
    row = info_file.iloc[i]
    data_path = Path('../assets/emg_20221112')
    name = row['name']
    date = datetime.strptime(row['time'], "%Y/%m/%d %H:%M").strftime('%Y-%m-%d')
    event_file = [event_file for event_file in (data_path / 'raw').glob(f'{name}*{date}*events*')]
    events_df = pd.read_csv(event_file[0])
    print(events_df)
# %%
for i in [8]:
    row = info_file.iloc[i]

    name = row['name']
    date = datetime.strptime(row['time'], "%Y/%m/%d %H:%M").strftime('%Y-%m-%d')

    curl_file_name = f'{name}_{date}_curl.csv'
    emg_file_name = f'{name}_{date}_emg_noabs.csv'

    curl_file = data_path / curl_file_name
    emg_file = data_path / emg_file_name

    curl_data = np.genfromtxt(curl_file, delimiter=',')[:-9000]
    # curl_data = np.genfromtxt(curl_file, delimiter=',')[:-32000]

    seg_start = []
    seg_end = []
    curl_df = pd.DataFrame(curl_data)
    des = curl_df.describe(percentiles=[0.05, 0.95])
    print(des)
    # for i in range(int(len(curl_data) / 50 / 6)):
    #     seg_start.append(i * 50 * 6 + 75)
    #     seg_end.append(i * 50 * 6 + 150)
    # for i in range(int(len(curl_data) / 50 / 6)):
    #     print(np.sum(curl_data[seg_start[i]:seg_end[i]], axis=0) / 75)

# %%

for i in range(30):
    row = info_file.iloc[i]

    name = row['name']
    date = datetime.strptime(row['time'], "%Y/%m/%d %H:%M").strftime('%Y-%m-%d')

    curl_file_name = f'{name}_{date}_curl.csv'

    curl_file = data_path / curl_file_name
    curl_data = np.genfromtxt(curl_file, delimiter=',')[:-9000]
    curl_data[curl_data == 0] = np.nan
    curl_df = pd.DataFrame(curl_data)
    des = curl_df.describe(percentiles=[0.05, 0.2, 0.40, 0.80, 0.85, 0.9, 0.95])
    curl_df = pd.DataFrame(curl_data).clip([des[finger]['5%'] for finger in range(5)],
                                           [des[finger]['95%'] for finger in range(5)])
    bins = [pd.cut(curl_df[col], 16, labels=False, retbins=True)[1] for col in curl_df]
    curl_df = pd.DataFrame([pd.cut(curl_df[col], 16, labels=False) for col in curl_df]).transpose()
    plt.plot(curl_df[:5000])
    plt.show()
    plt.plot

    for idx in range(5):
        plt.hist(curl_df.to_numpy()[:, idx], bins=16, )
        plt.axvline((des[idx]["90%"] - des[idx]["5%"]) / (des[idx]["95%"] - des[idx]["5%"]) * 15,
                    color='g')
        plt.show()

    curl_df.to_csv(data_path / f'{name}_{date}_curl_calibrated.csv', header=False, index=False)
