import pandas as pd

def sort_dem(csv_path):
  df = pd.read_csv(csv_path)
  df = df.sort_values(['latitude', 'longitude'])
  output_path = csv_path.split('.csv')[0]
  df.to_csv(f'{output_path}_sorted.csv', index=False)

if __name__ == '__main__':
  path = './csv/DEM/IzuOshima10m.csv'
  sort_dem(path)