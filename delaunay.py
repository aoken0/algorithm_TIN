import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

def delaunay_triangulation(df_p: pd.DataFrame | np.ndarray, point=None, p_num=None, plot=False, return_height=False, return_adjacent_points=False, output_name=None, show_num=False, return_triangles=False):
  # output_dir = './output'
  height = None
  points = None
  data = None

  if (type(df_p) == pd.DataFrame):
    points = df_p[['x', 'y']].to_numpy()
    data = df_p[['x', 'y', 'h']].to_numpy()
  elif (type(df_p) == np.ndarray):
    points = df_p[:, :2]
    data = df_p

  # Delaunay 三角形分割を計算
  tri = Delaunay(points)

  # 点が指定されていたらどこに内包されているか
  if point is not None:
    p = tri.find_simplex(point)
    # print(tri.simplices[p]) # 内包されている三角形の頂点番号
    triangle_points = data[tri.simplices[p]]
    if return_height:
      height = get_height(triangle_points=triangle_points, point=point)

  # plotモードのとき
  if plot:
    plot_network(points=points, df_p=df_p, tri=tri, point=point, output_name=output_name)
  
  return_val = []

  if return_height:
    return_val.append(height)
  
  if return_adjacent_points:
    return_val.append(get_adjacent_points(tri, p_num))

  if return_triangles:
    return_val.append(tri.simplices)
  
  return return_val


def get_adjacent_points(tri: Delaunay, point):
  # print(tri.simplices)
  # print(point)
  points = [triangle for triangle in tri.simplices if point in triangle]
  points = np.array(points)
  points = points.flatten()
  points = np.unique(points)
  points = points[points != point]
  # print(points)
  return points


def plot_network(points, df_p: pd.DataFrame, tri, point=None, output_name=None, show_num=False):
  # 可視化
  plt.figure(figsize=(8, 6))

  # 三角形を描画
  for simplex in tri.simplices:
    x = np.append(points[simplex, 0], points[simplex, 0][0])
    y = np.append(points[simplex, 1], points[simplex, 1][0])
    plt.plot(x, y, 'k-')

  # 点群をプロット
  sc = plt.scatter(points[:, 0], points[:, 1], c=df_p['h'], cmap='viridis', marker='o', s=50)
  plt.colorbar(sc)

  # 頂点番号を表示
  if show_num:
    for i, row in df_p.iterrows():
      id, x, y = row['id'], row['x'], row['y']
      id = int(id)
      plt.text(x, y, f"{id}", color="blue", fontsize=10)
  
  if point is not None:
    plt.plot(point[0], point[1], 'ro')

  plt.gca().set_aspect('equal')
  if output_name is not None:
    plt.savefig(f'{output_name}.png')
  plt.show()



def get_height(triangle_points, point):
  vector = np.zeros((2, 2)) # ベクトルを格納する用
  
  # 三角形の全体の面積
  vector[0] = get_vector(triangle_points[0][0:2], triangle_points[1][0:2])
  vector[1] = get_vector(triangle_points[0][0:2], triangle_points[2][0:2])
  full_area = 0.5 * np.cross(vector[0], vector[1])

  # 内包された点と他2点から成る三角形の面積を3通り求める
  a0 = a1 = a2 = 0
  # 点0, 1 -> a2
  vector[0] = get_vector(triangle_points[0][0:2], triangle_points[1][0:2])
  vector[1] = get_vector(triangle_points[0][0:2], point)
  t_area = 0.5 * np.cross(vector[0], vector[1])
  a2 = t_area / full_area

  # 点1, 2 -> a0
  vector[0] = get_vector(triangle_points[1][0:2], triangle_points[2][0:2])
  vector[1] = get_vector(triangle_points[1][0:2], point)
  t_area = 0.5 * np.cross(vector[0], vector[1])
  a0 = t_area / full_area

  # 点2, 0 -> a1
  vector[0] = get_vector(triangle_points[2][0:2], triangle_points[0][0:2])
  vector[1] = get_vector(triangle_points[2][0:2], point)
  t_area = 0.5 * np.cross(vector[0], vector[1])
  a1 = t_area / full_area

  height = a0 * triangle_points[0][2] + a1 * triangle_points[1][2] + a2 * triangle_points[2][2]

  return height

def get_vector(point1, point2):
  return point1 - point2


if __name__ == '__main__':
  df_p = pd.read_csv('./csv/grid_400points.csv')
  # print(df_p)
  point = [50, 50]
  delaunay_triangulation(df_p, plot=True)