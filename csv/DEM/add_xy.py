import pandas as pd
import numpy as np

def add_xy(path, x_step, y_step, row, col):
  df = pd.read_csv(path)

  x = np.arange(col) * x_step
  y = np.arange(row) * y_step
  
  xx, yy = np.meshgrid(x, y)
  df['x'] = xx.flatten()
  df['y'] = yy.flatten()

  df.to_csv(f'{path.split(".csv")[0]}_xy.csv', index=False)

if __name__ == '__main__':
  h = 10.16
  v = 12.33
  add_xy('./csv/DEM/IzuOshima10m_sorted.csv', h, v, 1094, 914)