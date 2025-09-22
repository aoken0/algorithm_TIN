import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import ctypes
c_lib = ctypes.cdll.LoadLibrary("./c/func.so")
c_lib.get_height.argtypes = [
  ctypes.POINTER(ctypes.c_double),
  ctypes.POINTER(ctypes.c_double)
]
c_lib.get_height.restype = ctypes.c_double

def delaunay_triangulation(p_all: pd.DataFrame | np.ndarray, p_target: np.ndarray=None, p_num=None, plot=False, return_height=False, output_name=None, show_num=False, return_triangles=False):
  # output_dir = './output'
  height = None
  points = None
  data = None
  heights = None

  if (type(p_all) == pd.DataFrame):
    points = p_all[['x', 'y']].to_numpy()
    data = p_all[['x', 'y', 'h']].to_numpy()
    heights = p_all['h'].to_numpy()
  elif (type(p_all) == np.ndarray):
    points = p_all[:, :2]
    data = p_all
    heights = p_all[:, 2]

  # Delaunay 三角形分割を計算
  tri = Delaunay(points)

  # 点が指定されていたらどこに内包されているか
  if p_target is not None:
    p = tri.find_simplex(p_target)
    # print(tri.simplices[p]) # 内包されている三角形の頂点番号
    triangle_points = data[tri.simplices[p]]
    if return_height:
      if p == -1:
        height = 2 ** 10
      else:
        triangle_points_ptr = triangle_points.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_target = p_target.copy()
        p_target_ptr = p_target.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        height = c_lib.get_height(triangle_points_ptr, p_target_ptr)

  # plotモードのとき
  if plot:
    plot_network(points=points, heights=heights, triangles=tri.simplices, p_target=p_target, output_name=output_name, show_num=show_num)
  
  return_val = []

  if return_height:
    return_val.append(height)

  if return_triangles:
    return_val.append(tri.simplices)
  
  return return_val


def plot_network(points, heights, triangles, p_target=None, output_name=None, show_num=False):
  # 可視化
  plt.figure(figsize=(8, 6))

  # 三角形を描画
  for tri in triangles:
    x = np.append(points[tri, 0], points[tri, 0][0])
    y = np.append(points[tri, 1], points[tri, 1][0])
    plt.plot(x, y, 'k-')

  # 点群をプロット
  sc = plt.scatter(points[:, 0], points[:, 1], c=heights, cmap='viridis', marker='o', s=50)
  plt.colorbar(sc)

  # 頂点番号を表示
  if show_num:
    for id_, x, y in zip(p_all['id'], p_all['x'], p_all['y']):
      id = int(id_)
      plt.text(x, y, f"{id}", color="blue", fontsize=10)

  if p_target is not None:
    plt.plot(p_target[0], p_target[1], 'ro')

  plt.gca().set_aspect('equal')
  if output_name is not None:
    plt.savefig(f'{output_name}.png')
  plt.show()



def get_height(triangle_points, p_target):
  vector = np.zeros((2, 2)) # ベクトルを格納する用
  
  # 三角形の全体の面積
  vector[0] = get_vector(triangle_points[0][0:2], triangle_points[1][0:2])
  vector[1] = get_vector(triangle_points[0][0:2], triangle_points[2][0:2])
  full_area = 0.5 * np.cross(vector[0], vector[1])

  # 内包された点と他2点から成る三角形の面積を3通り求める
  a0 = a1 = a2 = 0
  # 点0, 1 -> a2
  vector[0] = get_vector(triangle_points[0][0:2], triangle_points[1][0:2])
  vector[1] = get_vector(triangle_points[0][0:2], p_target)
  t_area = 0.5 * np.cross(vector[0], vector[1])
  a2 = t_area / full_area

  # 点1, 2 -> a0
  vector[0] = get_vector(triangle_points[1][0:2], triangle_points[2][0:2])
  vector[1] = get_vector(triangle_points[1][0:2], p_target)
  t_area = 0.5 * np.cross(vector[0], vector[1])
  a0 = t_area / full_area

  # 点2, 0 -> a1
  vector[0] = get_vector(triangle_points[2][0:2], triangle_points[0][0:2])
  vector[1] = get_vector(triangle_points[2][0:2], p_target)
  t_area = 0.5 * np.cross(vector[0], vector[1])
  a1 = t_area / full_area

  height = a0 * triangle_points[0][2] + a1 * triangle_points[1][2] + a2 * triangle_points[2][2]

  return height

def get_vector(point1, point2):
  return point2 - point1


if __name__ == '__main__':
  p_all = pd.read_csv('./csv/grid_400points.csv')
  # print(p_all)
  p_target = [50, 50]
  delaunay_triangulation(p_all, plot=True)