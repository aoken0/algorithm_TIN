import pandas as pd
import numpy as np

def add_latlon(reference_file, output_file):
  df_ref = pd.read_csv(reference_file)
  df_ref['x'] = np.round(df_ref['x'], 2)
  df_ref['y'] = np.round(df_ref['y'], 2)

  df_out = pd.read_csv(output_file, keep_default_na=False)
  
  df_out = df_out.astype({'x': 'float64', 'y': 'float64'})
  df_out['latitude'] = np.nan
  df_out['longitude'] = np.nan

  for i in range(len(df_out)):
    x = float(df_out.iloc[i, 0])
    if np.isnan(x): continue
    y = float(df_out.iloc[i, 1])

    matched_data = df_ref.loc[(df_ref['x'] == x) & (df_ref['y'] == y)]
    df_out.iloc[i, 3] = matched_data['latitude'].values[0]
    df_out.iloc[i, 4] = matched_data['longitude'].values[0]

    if i % 1000 == 0:
      print(i)

  output_name = output_file.split('.csv')[0] + '_latlon.csv'
  df_out.to_csv(output_name, index=False, na_rep='nan')


if __name__ == '__main__':
  reference = './csv/DEM/IzuOshima10m_sorted_xy.csv'
  output = './output/DEM/IzuOshima10m_sorted_xy/1.0/points.csv'
  add_latlon(reference, output)
