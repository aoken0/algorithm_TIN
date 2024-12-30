from delaunay import delaunay_triangulation
from scipy.spatial import KDTree
import pandas as pd
import numpy as np
import time
import os

def grid_to_TIN_3(df_p: pd.DataFrame, row=50, col=50, x_step=1, y_step=1, max_error=0.05, output_name=None):
  start = time.time() # 実行時間計算用
  p_amount = len(df_p) # 点の数
  error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
  points = df_p[['x', 'y', 'h']].to_numpy() # 頂点情報をndarrayに変換
  points_reshape = np.reshape(points, (row, col, 3)) # 3次元配列に変換

  # ============================================================
  # ステップ 1: すべての点について標高誤差を計算
  # ============================================================  
  s1 = time.time()
  for i in range(points.shape[0]):
    point = points[i, :2]
    [x, y] = point[0:2]
    [x, y] = [int(x / x_step), int(y / y_step)]
    [r_start, r_end] = [max(0, y-3), min(row, y+4)]
    [c_start, c_end] = [max(0, x-3), min(col, x+4)]
    tmp_points = np.reshape(points_reshape[r_start:r_end, c_start:c_end], (-1, 3))
    tmp_points = tmp_points[np.any(tmp_points != points[i, :], axis=1), :]
    height = delaunay_triangulation(tmp_points, point, return_height=True)
    error[i] = abs(points[:, 2][i] - height)
  print(time.time() - s1)
  # return
  while 1:
    # ============================================================
    # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
    # ============================================================
    # print(error.min())
    if (error.min() > max_error): break

    # ============================================================
    # ステップ 3: 削除する点に隣接する点の誤差を初期化して削除
    # ============================================================
    del_id = error.argmin()
    # 削除する点に隣接(接続)する点とすべての三角形の情報を取得
    tri = delaunay_triangulation(points, p_num=del_id, return_adjacent_points=True, return_triangles=True)

    # 隣接点とすべての三角形の情報から削除点から3つ先までの点を抽出
    neighbor_points_index = []
    # 削除点の隣接点から2つ先までの点を抽出
    neighbor_points_index = get_neighbor_points(tri[0], tri[1], neighbor_points_index, 2)
    neighbor_points_index = np.array(neighbor_points_index)
    neighbor_points_index = neighbor_points_index[neighbor_points_index != del_id] # 削除点を除く

    # 隣接点における誤差を初期化
    error[tri[0]] = np.inf

    # ============================================================
    # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を再計算
    # ============================================================
    for i in tri[0]:
      point = points[i, :2]
      tmp_points_index = neighbor_points_index[neighbor_points_index != i]
      tmp_points = points[tmp_points_index]
      height = delaunay_triangulation(tmp_points, point, return_height=True)
      error[i] = abs(points[:, 2][i] - height)


    # 頂点情報から削除する点を削除
    points = np.delete(points, del_id, axis=0)
    # 標高誤差から削除する点を削除
    error = np.delete(error, del_id)
  
  print('fin')
  print(time.time() - start, 's')

  # output用のフォルダをなければ作成
  output_path = f'./output/{output_name}/{max_error}_3_test_1'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  triangles = delaunay_triangulation(points, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
  triangles = np.array(triangles[0])

  np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d')
  np.savetxt(f'{output_path}/TIN_points.csv', points, fmt='%f')





# def grid_to_TIN_4(df_p: pd.DataFrame, row=50, col=50, x_step=1, y_step=1, max_error=0.05, output_name=None):
#   start = time.time() # 実行時間計算用
#   p_amount = len(df_p) # 点の数
#   error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
#   points = df_p[['x', 'y', 'h']].to_numpy() # 頂点情報をndarrayに変換
#   points_reshape = np.reshape(points, (row, col, 3)) # 3次元配列に変換

#   s0 = s1 = s2 = s3 = t0 = 0.0

#   # ============================================================
#   # ステップ 1: すべての点について標高誤差を計算
#   # ============================================================
#   s = time.time()
#   for i in range(points.shape[0]):
#     point = points[i, :2]
#     [x, y] = point[0:2]
#     [x, y] = [int(x / x_step), int(y / y_step)]
#     [r_start, r_end] = [max(0, y-2), min(row, y+3)]
#     [c_start, c_end] = [max(0, x-2), min(col, x+3)]
#     tmp_points = np.reshape(points_reshape[r_start:r_end, c_start:c_end], (-1, 3))
#     tmp_points = tmp_points[np.any(tmp_points != points[i, :], axis=1), :]
#     height = delaunay_triangulation(tmp_points, point, return_height=True)
#     error[i] = abs(points[:, 2][i] - height)
#   s0 += time.time() - s

#   while 1:
#     # ============================================================
#     # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
#     # ============================================================
#     # print(error.min())
#     if (error.min() > max_error): break

#     # ============================================================
#     # ステップ 3: 削除する点に隣接する点の誤差を初期化して削除
#     # ============================================================
#     s = time.time()
#     del_id = error.argmin()

#     # 削除点から3つ先までの点になる可能性のある点をKDTreeを用いて求める
#     # つまり，削除点から近傍の49点を取得(削除点含む)
#     tree = KDTree(points[:, :2])
#     center = points[del_id, :2]
#     _, indices = tree.query(center, k=150)
#     indices = np.sort(indices)
#     tmp_points = points[indices]
#     tmp_del_id = np.where(indices == del_id)[0][0]
#     # 削除する点に隣接(接続)する点とすべての三角形の情報を取得
#     tri = delaunay_triangulation(tmp_points, p_num=tmp_del_id, return_adjacent_points=True, return_triangles=True)

#     t0 += time.time() - s

#     s = time.time()
#     adjacent_points = indices[tri[0]]

#     # 隣接点とすべての三角形の情報から削除点から3つ先までの点を抽出
#     neighbor_points_index = []
#     # 削除点の隣接点から2つ先までの点を抽出
#     neighbor_points_index = get_neighbor_points(tri[0], tri[1], neighbor_points_index, 2)
#     neighbor_points_index = np.array(neighbor_points_index)
#     neighbor_points_index = neighbor_points_index[neighbor_points_index != tmp_del_id] # 削除点を除く
#     neighbor_points_index = np.sort(neighbor_points_index)
#     neighbor_points_index = indices[neighbor_points_index]
#     s1 += time.time() - s

#     # 隣接点における誤差を初期化
#     error[tri[0]] = np.inf

#     # ============================================================
#     # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を再計算
#     # ============================================================
#     s = time.time()
#     for i in adjacent_points:
#       point = points[i, :2]
#       tmp_points_index = neighbor_points_index[neighbor_points_index != i]
#       tmp_points = points[tmp_points_index]
#       height = delaunay_triangulation(tmp_points, point, return_height=True)
#       error[i] = abs(points[:, 2][i] - height)
#     s2 += time.time() - s

#     s = time.time()
#     # 頂点情報から削除する点を削除
#     points = points[np.any(points != points[del_id, :], axis=1), :]
#     # 標高誤差から削除する点を削除
#     error = np.delete(error, del_id)
#     s3 += time.time() - s
  
#   print('fin')
#   print(time.time() - start, 's')
#   print(s0, t0, s1, s2, s3)

#   # ============================================================
#   # 出力等
#   # ============================================================
#   # output用のフォルダをなければ作成
#   output_path = f'./output/{output_name}/{max_error}_3_test'
#   if not os.path.exists(output_path):
#     os.makedirs(output_path)
#   triangles = delaunay_triangulation(points, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
#   triangles = np.array(triangles[0])
#   np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d')
#   np.savetxt(f'{output_path}/TIN_points.csv', points, fmt='%f')


def grid_to_TIN_4(df_p: pd.DataFrame, row=50, col=50, x_step=1, y_step=1, max_error=0.05, output_name=None):
  start = time.time() # 実行時間計算用
  p_amount = len(df_p) # 点の数
  error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
  points = df_p[['x', 'y', 'h']].to_numpy() # 頂点情報をndarrayに変換
  points_reshape = np.reshape(points, (row, col, 3)) # 3次元配列に変換
  # 削除した点の数
  del_amount = 0

  # s0 = s1 = s2 = s3 = t0 = 0.0

  # ============================================================
  # ステップ 1: すべての点について標高誤差を計算
  # ============================================================
  # s = time.time()
  for i in range(points.shape[0]):
    point = points[i, :2]
    [x, y] = point[0:2]
    [x, y] = [int(x / x_step), int(y / y_step)]
    [r_start, r_end] = [max(0, y-2), min(row, y+3)]
    [c_start, c_end] = [max(0, x-2), min(col, x+3)]
    tmp_points = np.reshape(points_reshape[r_start:r_end, c_start:c_end], (-1, 3))
    tmp_points = tmp_points[np.any(tmp_points != points[i, :], axis=1), :]
    height = delaunay_triangulation(tmp_points, point, return_height=True)
    error[i] = abs(points[:, 2][i] - height)
  # s0 += time.time() - s

  while 1:
    # ============================================================
    # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
    # ============================================================
    # print(error.min())
    if (error.min() > max_error): break
    del_amount += 1
    if del_amount % 1000 == 0:
      print(del_amount)

    # ============================================================
    # ステップ 3: 削除する点に隣接する点の誤差を初期化して削除
    # ============================================================
    # s = time.time()
    del_id = error.argmin()

    # 削除点から3つ先までの点になる可能性のある点をKDTreeを用いて求める
    # つまり，削除点から近傍の49点を取得(削除点含む)
    tree = KDTree(points[:, :2])
    center = points[del_id, :2]
    _, indices = tree.query(center, k=100)
    indices = np.sort(indices)
    filtered_points = points[indices]
    tmp_del_id = np.where(indices == del_id)[0][0]
    # 削除する点に隣接(接続)する点とすべての三角形の情報を取得
    tri = delaunay_triangulation(filtered_points, p_num=tmp_del_id, return_adjacent_points=True, return_triangles=True)

    # t0 += time.time() - s

    # s = time.time()
    # 隣接点とすべての三角形の情報から削除点から3つ先までの点を抽出
    neighbor_points_index = []
    # 削除点の隣接点から2つ先までの点を抽出
    neighbor_points_index = get_neighbor_points(tri[0], tri[1], neighbor_points_index, 2)
    neighbor_points_index = np.array(neighbor_points_index)
    neighbor_points_index = neighbor_points_index[neighbor_points_index != tmp_del_id] # 削除点を除く
    neighbor_points_index = np.sort(neighbor_points_index)
    neighbor_points_index = indices[neighbor_points_index]
    adjacent_points = indices[tri[0]]
    # s1 += time.time() - s

    # 隣接点における誤差を初期化
    error[adjacent_points] = np.inf

    # ============================================================
    # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を再計算
    # ============================================================
    # s = time.time()
    for i in adjacent_points:
      point = points[i, :2]
      tmp_points_index = neighbor_points_index[neighbor_points_index != i]
      tmp_points = points[tmp_points_index]
      height = delaunay_triangulation(tmp_points, point, return_height=True)
      error[i] = abs(points[:, 2][i] - height)
    # s2 += time.time() - s

    # s = time.time()
    # 頂点情報から削除する点を削除
    points = points[np.any(points != points[del_id, :], axis=1), :]
    # 標高誤差から削除する点を削除
    error = np.delete(error, del_id)
    # s3 += time.time() - s
  
  print('fin')
  print(time.time() - start, 's')
  # print(s0, t0, s1, s2, s3)

  # ============================================================
  # 出力等
  # ============================================================
  # output用のフォルダをなければ作成
  output_path = f'./output/{output_name}/{max_error}'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  triangles = delaunay_triangulation(points, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
  triangles = np.array(triangles[0])
  np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d')
  np.savetxt(f'{output_path}/TIN_points.csv', points, fmt='%f')


def get_neighbor_points(reference_points: list, triangles: np.ndarray, neighbor_points_index: list, depth: int):
  for i in range(depth):
    for num in reference_points:
      tmp = triangles[np.any(triangles == num, axis=1), :]
      tmp = np.reshape(tmp, (-1, 3))
      tmp = np.unique(tmp)
      for num2 in tmp:
        if num2 in neighbor_points_index: continue
        neighbor_points_index.append(num2)
    
    if i > depth - 2: break
    reference_points = list(set(neighbor_points_index) - set(reference_points))
      
  return neighbor_points_index

if __name__ == '__main__':
  folder = './csv'
  # filename = 'grid_1000000points_shift.csv'
  # [row, col] = [1000, 1000]
  # filename = 'grid_22500points_shift.csv'
  # [row, col] = [150, 150]
  # filename = 'grid_10000points_shift.csv'
  # [row, col] = [100, 100]
  # filename = 'grid_2500points_shift.csv'
  # [row, col] = [50, 50]
  # filename = 'grid_400points_shift.csv'
  # [row, col] = [20, 20]
  filename = '/DEM/IzuOshima10m_sorted_xy.csv'
  [row, col] = [1094, 914]
  x_step, y_step = 10.16, 12.33
  max_error = 1.0
  output_name = filename.split('.')[0]
  df_p = pd.read_csv(f'{folder}/{filename}')
  # grid_to_TIN_3(df_p, row=row, col=col, x_step=1, y_step=1.1, max_error=max_error, output_name=output_name)
  grid_to_TIN_4(df_p, row=row, col=col, x_step=x_step, y_step=y_step, max_error=max_error, output_name=output_name)
