import pandas as pd
import numpy as np

def extract_edge(path, filename):
  df_t = pd.read_csv(f'{path}/{filename}', header=None)

  triangles = df_t.to_numpy()
  edges = np.vstack([
    triangles[:, [0, 1]],
    triangles[:, [1, 2]],
    triangles[:, [2, 0]]
  ])
  sorted_edges = np.sort(edges, axis=1)
  unique_edges = np.unique(sorted_edges, axis=0)

  np.savetxt(f'{path}/edges.csv', unique_edges, delimiter=',', fmt='%d')
  
  e_quantity = len(unique_edges)
  return e_quantity
  

if __name__ == '__main__':
  path = "./output/DEM/IzuOshima10m_sorted_xy/2.0"
  filename = 'TIN_triangles.csv'
  extract_edge(path, filename)