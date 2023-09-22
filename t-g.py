import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import (
    AutoDateLocator,
    AutoDateFormatter
)
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


def parse_df(_df, _target, _csv):
    _date_start = '2023-08-30'
    _date_end = '2023-09-03'
    org_df_se = _df[_date_start:_date_end]
    print(org_df_se.head())
    print(org_df_se.tail())
    plt.figure(figsize=(24, 6))
    plt.plot(org_df_se[_target], color="black")
    
    ##### timestamp column is not evenly spaced and monotonically increasing we get an error when using PandasDataset. Here we show how to fill in the gaps that are missing
    max_end = max(_df.groupby("item_id").apply(lambda x: x.index[-1]))
    dfs_dict = {}
    for item_id, gdf in _df.groupby("item_id"):
        new_index = pd.date_range(gdf.index[0], end=max_end, freq="5T")  # 设置为5分钟频率
        # new_gdf = gdf.resample('5T').interpolate(method='linear')
        # new_gdf["item_id"] = item_id
        # dfs_dict[item_id] = new_gdf
        dfs_dict[item_id] = gdf.reindex(new_index).assign(item_id=item_id).fillna(method='ffill')  # "forward fill", 也可以使用fillna填充缺失值
    new_df = dfs_dict[0]
    new_df_se = new_df[_date_start:_date_end]
    print(new_df_se.head())
    print(new_df_se.tail())
    plt.plot(new_df_se[_target], color="red")

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data')
    plt.grid(True)
    # 自动设置日期时间刻度和格式
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)  # 旋转刻度标签以避免重叠
    plt.tight_layout()
    _fn, _ext = os.path.splitext(os.path.basename(_csv))
    plt.savefig(f"{_fn}_{_target}_{_date_start}_{_date_end}.png")

    # 这里dfs_dict包含了重新索引后的DataFrame
    _ds = PandasDataset(dfs_dict[0], target=_target)
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
        trainer_kwargs={"max_epochs": 5}
    ).train(training_data)
    forecasts = list(model.predict(test_data.input))
    # print(forecasts)
    ##### Plot predictions
    plt.plot(_df[_target], color="black")
    for forecast in forecasts:
        forecast.plot(show_label=True)
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
    plt.savefig(f"{_fn}_g_DeepAR_{_target}.png")



_csv = sys.argv[1]

# df = pd.read_csv(_csv)
# df.rename(columns={'timestamp': 'ds'}, inplace=True)
# df['ds'] = pd.to_datetime(df['ds'])
# print(df.head())

df = pd.read_csv(_csv, index_col=0, parse_dates=True)
print(df.head())

other_columns = df.columns[0:]
print(other_columns)

df["item_id"] = 0
df = df.sort_index(ascending=True)
print(df.head())
print(df.tail())
# exit()


for i in other_columns:
    print("----")
    print(i)
    print("----")
    parse_df(df[[i, "item_id"]], i, _csv)
    print("----")

