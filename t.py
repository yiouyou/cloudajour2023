import pandas as pd

# 创建示例 DataFrame
data = {'average': [89.544, 89.560, 89.517, 89.867, 89.659],
        'item_id': [0, 0, 0, 0, 0]}
date_rng = pd.date_range(start='2023-09-07 00:03:00', end='2023-09-07 23:56:00', freq='T')
df = pd.DataFrame(data, index=date_rng)

# 定义自定义插值函数
def custom_interpolate(group):
    return group.interpolate(method='linear')

# 使用resample方法和apply方法进行插值
resampled_df = df.resample('5T').apply(custom_interpolate)

print(resampled_df)
