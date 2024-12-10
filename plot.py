import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_points(df):
  plt.scatter(df['x'], df['y'])
  plt.xlabel('x')
  plt.ylabel('y')
  for id, x, y in zip(df['id'], df['x'], df['y']):
    plt.annotate(id, (x, y))
  plt.show()

def plot_points_edges(df_p, df_e):
  plt.scatter(df_p['x'], df_p['y'])
  plt.xlabel('x')
  plt.ylabel('y')
  for id, x, y in zip(df_p['id'], df_p['x'], df_p['y']):
    plt.annotate(id, (x, y))
  
  p1_list = df_e['p1']
  p2_list = df_e['p2']

  for p1, p2 in zip(p1_list, p2_list):
    x1, y1 = df_p[df_p['id'] == p1]['x'], df_p[df_p['id'] == p1]['y']
    x2, y2 = df_p[df_p['id'] == p2]['x'], df_p[df_p['id'] == p2]['y']
    plt.plot([x1, x2], [y1, y2], 'k-', ajustable='box')

  plt.show()

if __name__ == '__main__':
  df_p = pd.read_csv('./csv/random_points.csv')
  # plot_points(df_p)
  df_e = pd.read_csv('./csv/random_edges.csv')
  plot_points_edges(df_p, df_e)
