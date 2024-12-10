import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

def delaunay_triangulation(df_p, point):
  output_dir = './output'
  # print(df_p)
  points = df_p[['x', 'y']].to_numpy()

  # Delaunay 三角形分割を計算
  tri = Delaunay(points)

  # 可視化
  plt.figure(figsize=(8, 6))

  # 点群をプロット
  plt.plot(points[:, 0], points[:, 1], 'o')

  # 三角形を描画
  for simplex in tri.simplices:
    x = np.append(points[simplex, 0], points[simplex, 0][0])
    y = np.append(points[simplex, 1], points[simplex, 1][0])
    plt.plot(x, y, 'k-')

  # オプション: 頂点番号を表示
  for i, (x, y) in enumerate(points):
    plt.text(x, y, f"{i}", color="blue", fontsize=10)
  
  # 点が指定されていたらどこに内包されているか
  if point is not None:
    p = tri.find_simplex(point)
    print(tri.simplices[p]) # 内包されている三角形の頂点番号
    data = df_p[['x', 'y', 'h']].to_numpy()
    triangle_points = data[tri.simplices[p]]
    get_height(triangle_points=triangle_points, point=point)
    plt.plot(point[0], point[1], 'ro')

  # プロット
  plt.gca().set_aspect('equal')
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

# def delaunay_triangulation_old(df_p):
#   output_dir = './output'
#   x_points = df_p['x'].to_numpy()
#   y_points = df_p['y'].to_numpy()

#   tri_delaunay = tri.Triangulation(x=x_points, y=y_points)

#   print(tri_delaunay.triangles)
#   triangles = tri_delaunay.triangles
#   df_triangles = pd.DataFrame(triangles, columns=['p1', 'p2', 'p3'])
#   print(df_triangles)

#   plt.triplot(tri_delaunay)
#   plt.show()


if __name__ == '__main__':
  df_p = pd.read_csv('./csv/random_points.csv')
  print(df_p)
  point = [8, 5]
  delaunay_triangulation(df_p, point)