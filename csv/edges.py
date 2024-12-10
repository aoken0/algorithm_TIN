import pandas as pd
import numpy as np

def random_edges(id, n):
  edges = np.array(np.random.choice(id, (n, 2)))
  print(edges)
  for i, ele in enumerate(edges):
    while 1:
      if np.unique(ele).size < 2:
        ele = edges[i] = np.random.choice(id, 2)
        print(ele)
      else:
        break
  
  edges = np.sort(edges, axis=1)
  edges = np.unique(edges, axis=0)
  print(edges)
  return 


if __name__ == '__main__':
  df_p = pd.read_csv('./csv/random_points.csv')
  print(df_p)
  print(random_edges(df_p['id'], 6))
