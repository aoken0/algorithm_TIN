from remove import remove_sea
from extract import extract_edge

def save_info(path, p_quantity, e_quantity):
  title = ['点の数', '辺の数']
  info = [p_quantity, e_quantity]
  with open(f'{path}/info.txt', 'w', encoding='utf-8') as f:
    for t, i in zip(title, info):
      f.write(f'{t}: {i}\n')

if __name__ == '__main__':
  path = "./output/DEM/IzuOshima10m_sorted_xy/1.0"
  filenames = ['TIN_points.csv', 'TIN_triangles.csv']
  p_quantity = remove_sea(path, filenames)
  e_quantity = extract_edge(path, filenames[1])
  save_info(path, p_quantity, e_quantity)
