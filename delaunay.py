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

def delaunay_triangulation(p_all: np.ndarray, p_target: np.ndarray=None, plot=False, return_height=False, output_name=None, show_num=False, return_triangles=False):
  height = None
  points = p_all[:, :2]
  data = p_all
  heights = p_all[:, 2]

  # Delaunay 三角形分割を計算
  tri = Delaunay(points)

  # 点が指定されていたらどこに内包されているか
  if p_target is not None:
    p = tri.find_simplex(p_target)
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

  if p_target is not None:
    plt.plot(p_target[0], p_target[1], 'ro')

  plt.gca().set_aspect('equal')
  if output_name is not None:
    plt.savefig(f'{output_name}.png')
  plt.show()

if __name__ == '__main__':
  pass