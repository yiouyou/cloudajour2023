import pandas as pd
import matplotlib.pyplot as plt
import sys, os

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.util import to_pandas
from gluonts.torch import DeepAREstimator
from gluonts.dataset.util import to_pandas


def highlight_entry(entry, color):
    start = entry["start"]
    end = entry["start"] + len(entry["target"])
    plt.axvspan(start, end, facecolor=color, alpha=0.2)


def plot_dataset_splitting(original_dataset, training_dataset, test_pairs):
    for original_entry, train_entry in zip(original_dataset, training_dataset):
        to_pandas(original_entry).plot()
        highlight_entry(train_entry, "red")
        plt.legend(["sub dataset", "training dataset"], loc="upper left")
        # plt.show()
        plt.savefig(f"111.png")
    for original_entry in original_dataset:
        for test_input, test_label in test_pairs:
            to_pandas(original_entry).plot()
            highlight_entry(test_input, "green")
            highlight_entry(test_label, "blue")
            plt.legend(["sub dataset", "test input", "test label"], loc="upper left")
            # plt.show()
            plt.savefig(f"222.png")


def parse_df(_df, _target):
    ##### timestamp column is not evenly spaced and monotonically increasing we get an error when using PandasDataset. Here we show how to fill in the gaps that are missing
    max_end = max(_df.groupby("item_id").apply(lambda _df: _df.index[-1]))
    dfs_dict = {}
    for item_id, gdf in df.groupby("item_id"):
        new_index = pd.date_range(gdf.index[0], end=max_end, freq="5T")  # 设置为5分钟频率
        dfs_dict[item_id] = gdf.reindex(new_index).assign(item_id=item_id).fillna(method='ffill')  # 可以使用fillna填充缺失值
    print(dfs_dict)
    # 这里dfs_dict包含了重新索引后的DataFrame
    _ds = PandasDataset(dfs_dict, target=_target)
    print(_ds)
    # _ds = PandasDataset.from_long_dataframe(ds, target=_target, item_id="item_id")

    ##### Split the data for training and testing
    prediction_length = 1 * 24 * 12
    # training_data, test_gen = split(_ds, offset=-840)
    training_data, test_gen = split(_ds, date=pd.Period("2023-09-14 00:00:00", freq="5T"))
    test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=3)
    # plot_dataset_splitting(_ds, training_data, test_data)
    # ##### Train the model and make predictions
    model = DeepAREstimator(
        prediction_length=prediction_length,
        freq="5T",
        trainer_kwargs={"max_epochs": 10}
    ).train(training_data)
    forecasts = list(model.predict(test_data.input))
    # print(forecasts)
    ##### Plot predictions
    plt.plot(_df[_target], color="black")
    for forecast in forecasts:
        forecast.plot(show_label=True)
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
    name, extension = os.path.splitext(_csv)
    plt.savefig(f"{name}_g_{_target}.png")


def remove_timezone(timestamp):
    return timestamp.replace("Z", "")

_csv = sys.argv[1]

df = pd.read_csv(_csv)
df.rename(columns={'timestamp': 'ds'}, inplace=True)
df['ds'] = df['ds'].apply(remove_timezone)
print(df.head())

df = pd.read_csv(_csv, index_col=0, parse_dates=True)
print(df.head())

other_columns = df.columns[0:]
print(other_columns)

df["item_id"] = 0
df = df.sort_index(ascending=True)
print(df.head())

# exit()


for i in other_columns:
    print(i)
    parse_df(df, i)
    print("----")

