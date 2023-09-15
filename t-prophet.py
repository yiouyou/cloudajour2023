# https://facebook.github.io/prophet/docs/quick_start.html
from prophet import Prophet
import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
print(df.head())


m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
print(future.tail())

forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

fig1.savefig('t-prophet_fig1.png')
fig2.savefig('t-prophet_fig2.png')

