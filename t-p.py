from prophet import Prophet
import pandas as pd
import sys, os

_periods = 7

def parse_df(_df, _csv, _key, _periods):
    _date_start = '2023-09-07'
    _date_end = '2023-09-11'
    org_df_se = _df[_date_start:_date_end]
    print(org_df_se.head())
    print(org_df_se.tail())

    m = Prophet()
    m.fit(org_df_se)
    future = m.make_future_dataframe(periods=_periods)
    # print(future.tail())
    forecast = m.predict(future)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    _fn, _ext = os.path.splitext(os.path.basename(_csv))
    fig1.savefig(f"{_fn}_p_{_key}_{_date_start}_{_date_end}_fig1.png")
    fig2.savefig(f"{_fn}_p_{_key}_{_date_start}_{_date_end}_fig2.png")


_csv = sys.argv[1]

df = pd.read_csv(_csv)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
print(df.head())
df['ds'] = df['timestamp']
print(df.head())
df = df.set_index('timestamp')
print(df.head())

# df = pd.read_csv(_csv, index_col=0, parse_dates=True)
# print(df.head())

other_columns = df.columns[0:-1]
print(other_columns)

df = df.sort_index(ascending=True)
print(df.head())
print(df.tail())
# exit()


dfs = {}
for col in other_columns:
    dfs[col] = df[['ds', col]].rename(columns={col: 'y'})

for _key, _df in dfs.items():
    print(f"'{_key}':")
    # print(_df)
    parse_df(_df, _csv, _key, _periods)
    print("----")

