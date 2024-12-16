import pandas as pd
import numpy as np

# グリッドのサイズを設定
x_min, x_max, x_step = 0, 20, 1  # x方向の範囲と間隔
y_min, y_max, y_step = 0, 20, 1  # y方向の範囲と間隔

# x, y のグリッドを生成
x_values = np.arange(x_min, x_max, x_step)
y_values = np.arange(y_min, y_max, y_step)
x_grid, y_grid = np.meshgrid(x_values, y_values)

print(x_grid)

# height 属性を生成（ここでは例としてサイン波 + ランダムノイズ）
height = np.sin(x_grid / 2) + np.cos(y_grid / 2) + np.random.normal(scale=0.2, size=x_grid.shape)

# 負の値が出ないよう全体をシフト
height += abs(height.min())

# データをフラット化して1次元配列に
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()
height_flat = height.flatten()

# データを統合してDataFrameに変換
points_df = pd.DataFrame({
    'id': np.arange(x_max * y_max),
    'x': x_flat,
    'y': y_flat,
    'h': height_flat
})

# 結果の確認
# print(points_df)

points_df.to_csv(f'./csv/grid_{x_max*y_max}points.csv', index=False)

