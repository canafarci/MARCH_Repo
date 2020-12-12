import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "ETH-USD"
EPOCHS = 10
BATCH_SIZE = 64

ratio = "ETH-USD"

def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df.dropna(inplace=True)
            indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
            df = df[indices_to_keep].astype(np.float64)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sels = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

main_df = pd.DataFrame()


dataset = f"_datasets/_crypto/{ratio}.csv"

df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
df.rename(columns={"close": f"{ratio}_close",
          "volume": f"{ratio}_volume"}, inplace=True)

df.set_index("time", inplace=True)
df = df[[f"{ratio}_close", f"{ratio}_volume"]]

if len(main_df) == 0:
    main_df = df
else:
    main_df = main_df.join(df)

main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(
    -FUTURE_PERIOD_PREDICT)

model = tf.keras.models.load_model("_models/_training_checkpoints/RNN_Final-10-0.470.model")

def predict(data):
    if model.predict([data])[0][0]*10 >= 1:
        print(1)
    else:
        print(0)

predict(preprocess_df(main_df.head()))