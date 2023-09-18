from prophet import Prophet
import pandas as pd
import sys, os

_periods = 7

def parse_df(_df, _csv, _key, _periods):
    m = Prophet()
    m.fit(_df)
    future = m.make_future_dataframe(periods=_periods)
    # print(future.tail())
    forecast = m.predict(future)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    name, extension = os.path.splitext(_csv)
    fig1.savefig(f"{name}_p_{_key}_fig1.png")
    fig2.savefig(f"{name}_p_{_key}_fig2.png")

def remove_timezone(timestamp):
    return timestamp.replace("Z", "")

_csv = sys.argv[1]

df = pd.read_csv(_csv)
# print(df.head())
df.rename(columns={'timestamp': 'ds'}, inplace=True)
df['ds'] = df['ds'].apply(remove_timezone)
# print(df.head())

other_columns = df.columns[1:]

dfs = {}
for col in other_columns:
    dfs[col] = df[['ds', col]].rename(columns={col: 'y'})

for _key, _df in dfs.items():
    print(f"'{_key}':")
    # print(_df)
    parse_df(_df, _csv, _key, _periods)
    print("----")

