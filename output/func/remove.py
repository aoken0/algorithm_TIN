import pandas as pd
import numpy as np

def remove_sea(path, filenames):
  # fileはpointsとtrianglesの2つのみ
  df_p = pd.read_csv(f'{path}/{filenames[0]}', header=None)
  df_t = pd.read_csv(f'{path}/{filenames[1]}', header=None)

  p = df_p.to_numpy()
  t = df_t.to_numpy()
  
  del_p_indices = np.where(p[:, 2] < 0)[0]

  t_indices = ~np.any(np.isin(t, del_p_indices), axis=1)
  
  p[del_p_indices] = [np.nan, np.nan, np.nan]
  output_p = pd.DataFrame(p, columns=['x', 'y', 'h'])
  output_t = t[t_indices]

  output_p.to_csv(f'{path}/points.csv', index=False, na_rep='nan')
  np.savetxt(f'{path}/triangles.csv', output_t, delimiter=',', fmt='%d')

  p_quantity = len(p) - len(del_p_indices)
  return p_quantity



if __name__ == '__main__':
  path = "./output/DEM/IzuOshima10m_sorted_xy/2.0"
  filenames = ['TIN_points.csv', 'TIN_triangles.csv']
  remove_sea(path, filenames)