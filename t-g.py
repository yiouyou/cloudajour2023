import pandas as pd
import matplotlib.pyplot as plt
import sys, os

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator


def parse_df(_df, _target):
    dataset = PandasDataset(_df, target=_target)
    # Split the data for training and testing
    training_data, test_gen = split(dataset, offset=-36)
    test_data = test_gen.generate_instances(prediction_length=12, windows=3)
    # Train the model and make predictions
    model = DeepAREstimator(
        prediction_length=12,
        freq="M",
        trainer_kwargs={"max_epochs": 5}
    ).train(training_data)
    forecasts = list(model.predict(test_data.input))
    print(forecasts)
    # Plot predictions
    # plt.plot(df, color="black")
    for forecast in forecasts:
        forecast.plot()
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
    name, extension = os.path.splitext(_csv)
    plt.savefig(f"{name}_g_{_target}.png")

def remove_timezone(timestamp):
    return timestamp.replace("Z", "")

_csv = sys.argv[1]

df = pd.read_csv(_csv)
# print(df.head())
df.rename(columns={'timestamp': 'ds'}, inplace=True)
df['ds'] = df['ds'].apply(remove_timezone)
# print(df.head())
# df.index = pd.to_datetime(df[df.columns[0]])
df.index = pd.to_datetime(df.index)
# exit()

other_columns = df.columns[1:]

dfs = {}
for col in other_columns:
    dfs[col] = df[['ds', col]]

for _key, _df in dfs.items():
    print(f"'{_key}':")
    # print(_df)
    _df.index = pd.to_datetime(_df.index)
    print(_df.head())
    print(_df.tail())
    parse_df(_df, _key)
    print("----")

