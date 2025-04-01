from matplotlib import pyplot as plt
from delaunay import delaunay_triangulation, plot_network
from scipy.spatial import KDTree
import pandas as pd
import numpy as np
import time
import os

def grid_to_TIN(df_p: pd.DataFrame, row=50, col=50, x_step=1, y_step=1, max_error=0.05, output_name=None):
  start = time.time() # 実行時間計算用
  p_amount = len(df_p) # 点の数
  error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
  p_all = df_p[['x', 'y', 'h']].to_numpy() # 頂点情報をndarrayに変換
  points_reshape = np.reshape(p_all, (row, col, 3)) # 3次元配列に変換
  # 削除した点の数
  del_amount = 0

  # ============================================================
  # ステップ 1: すべての点について標高誤差を計算
  # ============================================================
  for i in range(p_all.shape[0]):
    point = p_all[i, :2]
    [x, y] = point[0:2]
    [x, y] = [int(x / x_step), int(y / y_step)]
    [r_start, r_end] = [max(0, y-2), min(row, y+3)]
    [c_start, c_end] = [max(0, x-2), min(col, x+3)]
    tmp_points = np.reshape(points_reshape[r_start:r_end, c_start:c_end], (-1, 3))
    tmp_points = tmp_points[np.any(tmp_points != p_all[i, :], axis=1), :]
    height = delaunay_triangulation(tmp_points, point, return_height=True)
    error[i] = abs(p_all[:, 2][i] - height)

  # ドローネ分割を行い、すべての三角形を取得
  triangle_all = delaunay_triangulation(p_all, return_triangles=True)[0]

  while 1:
    # ============================================================
    # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
    # ============================================================
    if (error.min() > max_error): break
    del_amount += 1
    if del_amount % 1000 == 0:
      print(del_amount)

    # ============================================================
    # ステップ 3: 削除する点に隣接する点の誤差を初期化して削除
    # ============================================================
    del_id = error.argmin()

    # 削除点から3つ先までの点を取得する
    neighbor_point_indexes = get_neighbor_points(del_id, triangle_all, depth=3)
    p_adjacent = neighbor_point_indexes[0]
    p_neighbor_depth2 = neighbor_point_indexes[1]
    p_neighbor = neighbor_point_indexes[2]

    # ============================================================
    # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を再計算
    # ============================================================
    # 隣接点における誤差を初期化
    error[p_adjacent] = np.inf
    # 削除点に隣接する点全てにおいて削除して誤差計算する
    for i in p_adjacent:
      p_del = p_all[i, :2] # 隣接点の抽出(削除して誤差を求める対象点)
      p_tmp = p_neighbor[p_neighbor != i]
      p_tmp_neighbor = p_all[p_tmp]
      height = delaunay_triangulation(p_tmp_neighbor, p_del, return_height=True)
      error[i] = abs(p_all[:, 2][i] - height)

    # ============================================================
    # 削除点を除いたdelaunayネットワークを再構築
    # ============================================================
    # 深さ3の隣接点までを用いて計算する
    # 取得したネットワークの点番号は0から振られている => それをもとに戻す処理を行う
    new_triangle_local = delaunay_triangulation(p_all[p_neighbor_depth2], return_triangles=True)[0]
    new_triangle = p_neighbor_depth2[new_triangle_local] # ローカルの点番号をもともとの点番号に対応させる
    # 削除点と隣接点からなる三角形部分のみ更新する
    p_neighbor_depth2 = np.sort(np.array(p_neighbor_depth2))
    filtered_new_triangle = new_triangle[np.all(np.isin(new_triangle, p_adjacent), axis=1)]
    # 削除点と隣接点のみで構成された三角形を古いネットワークから取得
    p_del_neighbor = np.append(p_adjacent, del_id)
    filtered_triangle = triangle_all[~np.all(np.isin(triangle_all, p_del_neighbor), axis=1)]
    triangle_all = np.concatenate([filtered_triangle, filtered_new_triangle], 0)
   
    # 頂点情報から削除する点を削除
    p_all = p_all[np.any(p_all != p_all[del_id, :], axis=1), :]
    # 標高誤差から削除する点を削除
    error = np.delete(error, del_id)
    # s3 += time.time() - s
    # 三角形の頂点番号を更新
    triangle_all = np.where(triangle_all > del_id, triangle_all - 1, triangle_all)

  print('fin')
  print(time.time() - start, 's')

  # ============================================================
  # 出力等
  # ============================================================
  # output用のフォルダをなければ作成
  output_path = f'./output/{output_name}/{max_error}'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  triangles = delaunay_triangulation(p_all, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
  triangles = np.array(triangles[0])
  # plot_network(p_all[:, :2], p_all[:, 2], triangle_all, output_name=f'{output_path}/TIN')
  np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d', delimiter=',')
  np.savetxt(f'{output_path}/TIN_points.csv', p_all, fmt='%f', delimiter=',') 


def get_neighbor_points(p_reference: int, triangles: np.ndarray, depth: int):
  p_adjacent = []
  p_neighbor_depth2 = []
  p_neighbor = []
  p_searched = list([])
  reference_points = list([p_reference])
  for i in range(depth):
    for num in reference_points:
      tmp = triangles[np.any(triangles == num, axis=1), :]
      tmp = np.reshape(tmp, (-1, 3))
      tmp = np.unique(tmp)
      if (i == 0): p_adjacent = tmp[tmp != p_reference]
      for num2 in tmp:
        if num2 in p_neighbor: continue
        p_neighbor.append(num2)
    if (i == 1):
      p_tmp = np.array(p_neighbor)
      p_neighbor_depth2 = p_tmp[p_tmp != p_reference]
    if i > depth - 2: break
    p_searched.append(reference_points)
    reference_points = list(set(p_neighbor) - set(reference_points))
  # 基準点削除
  tmp = np.array(p_neighbor)
  p_neighbor = tmp[tmp != p_reference]
  return [p_adjacent, p_neighbor_depth2, p_neighbor]

if __name__ == '__main__':
  folder = './csv'
  # filename = 'grid_1000000points_shift.csv'
  # [row, col] = [1000, 1000]
  # filename = 'grid_22500points_shift.csv'
  # [row, col] = [150, 150]
  # filename = 'grid_10000points_shift.csv'
  # [row, col] = [100, 100]
  # filename = 'test/grid_2500points_shift.csv'
  # [row, col] = [50, 50]
  filename = 'test/grid_400points_shift.csv'
  [row, col] = [20, 20]
  x_step, y_step = 1.0, 1.1
  # filename = '/DEM/IzuOshima10m_sorted_xy.csv'
  # [row, col] = [1094, 914]
  # x_step, y_step = 10.16, 12.33
  max_error = 0.1
  output_name = filename.split('.')[0]
  df_p = pd.read_csv(f'{folder}/{filename}')
  grid_to_TIN(df_p, row=row, col=col, x_step=x_step, y_step=y_step, max_error=max_error, output_name=output_name)







# def get_neighbor_points(reference_points: list, triangles: np.ndarray, neighbor_point_indexes: list, depth: int):
#   for i in range(depth):
#     for num in reference_points:
#       tmp = triangles[np.any(triangles == num, axis=1), :]
#       tmp = np.reshape(tmp, (-1, 3))
#       tmp = np.unique(tmp)
#       for num2 in tmp:
#         if num2 in neighbor_point_indexes: continue
#         neighbor_point_indexes.append(num2)
    
#     if i > depth - 2: break
#     reference_points = list(set(neighbor_point_indexes) - set(reference_points))
      
#   return neighbor_point_indexes

# def grid_to_TIN(df_p: pd.DataFrame, row=50, col=50, x_step=1, y_step=1, max_error=0.05, output_name=None):
#   start = time.time() # 実行時間計算用
#   p_amount = len(df_p) # 点の数
#   error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
#   p_all = df_p[['x', 'y', 'h']].to_numpy() # 頂点情報をndarrayに変換
#   points_reshape = np.reshape(p_all, (row, col, 3)) # 3次元配列に変換
#   # 削除した点の数
#   del_amount = 0

#   # s0 = s1 = s2 = s3 = t0 = 0.0

#   # ============================================================
#   # ステップ 1: すべての点について標高誤差を計算
#   # ============================================================
#   # s = time.time()
#   for i in range(p_all.shape[0]):
#     point = p_all[i, :2]
#     [x, y] = point[0:2]
#     [x, y] = [int(x / x_step), int(y / y_step)]
#     [r_start, r_end] = [max(0, y-2), min(row, y+3)]
#     [c_start, c_end] = [max(0, x-2), min(col, x+3)]
#     tmp_points = np.reshape(points_reshape[r_start:r_end, c_start:c_end], (-1, 3))
#     tmp_points = tmp_points[np.any(tmp_points != p_all[i, :], axis=1), :]
#     height = delaunay_triangulation(tmp_points, point, return_height=True)
#     error[i] = abs(p_all[:, 2][i] - height)
#   # s0 += time.time() - s

#   while 1:
#     # ============================================================
#     # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
#     # ============================================================
#     # print(error.min())
#     if (error.min() > max_error): break
#     del_amount += 1
#     if del_amount % 1000 == 0:
#       print(del_amount)

#     # ============================================================
#     # ステップ 3: 削除する点に隣接する点の誤差を初期化して削除
#     # ============================================================
#     # s = time.time()
#     del_id = error.argmin()


#     # --------------------------------------------------------------------------------------
#     # --------------------------------------------------------------------------------------
#     # --------------------------------------------------------------------------------------
#     # 削除点から3つ先までの点になる可能性のある点をKDTreeを用いて求める
#     # つまり，削除点から近傍の49点を取得(削除点含む)
#     tree = KDTree(p_all[:, :2])
#     center = p_all[del_id, :2]
#     _, indices = tree.query(center, k=100)
#     indices = np.sort(indices)
#     filtered_points = p_all[indices]
#     tmp_del_id = np.where(indices == del_id)[0][0]
#     # 削除する点に隣接(接続)する点とすべての三角形の情報を取得
#     tri = delaunay_triangulation(filtered_points, p_num=tmp_del_id, return_adjacent_points=True, return_triangles=True)

#     # t0 += time.time() - s

#     # s = time.time()
#     # 隣接点とすべての三角形の情報から削除点から3つ先までの点を抽出
#     neighbor_point_indexes = []
#     # 削除点の隣接点から2つ先までの点を抽出
#     neighbor_point_indexes = get_neighbor_points(tri[0], tri[1], neighbor_point_indexes, 2)
#     neighbor_point_indexes = np.array(neighbor_point_indexes)
#     neighbor_point_indexes = neighbor_point_indexes[neighbor_point_indexes != tmp_del_id] # 削除点を除く
#     neighbor_point_indexes = np.sort(neighbor_point_indexes)
#     neighbor_point_indexes = indices[neighbor_point_indexes]
#     adjacent_points = indices[tri[0]]
#     # s1 += time.time() - s

#     # 隣接点における誤差を初期化
#     error[adjacent_points] = np.inf

#     # ============================================================
#     # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を再計算
#     # ============================================================
#     # s = time.time()
#     for i in adjacent_points:
#       point = p_all[i, :2]
#       tmp_points_index = neighbor_point_indexes[neighbor_point_indexes != i]
#       tmp_points = p_all[tmp_points_index]
#       height = delaunay_triangulation(tmp_points, point, return_height=True)
#       error[i] = abs(p_all[:, 2][i] - height)
#     # s2 += time.time() - s

#     # s = time.time()
#     # 頂点情報から削除する点を削除
#     p_all = p_all[np.any(p_all != p_all[del_id, :], axis=1), :]
#     # 標高誤差から削除する点を削除
#     error = np.delete(error, del_id)
#     # s3 += time.time() - s
  
#   print('fin')
#   print(time.time() - start, 's')
#   # print(s0, t0, s1, s2, s3)

#   # ============================================================
#   # 出力等
#   # ============================================================
#   # output用のフォルダをなければ作成
#   output_path = f'./output/{output_name}/{max_error}'
#   if not os.path.exists(output_path):
#     os.makedirs(output_path)
#   triangles = delaunay_triangulation(p_all, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
#   triangles = np.array(triangles[0])
#   np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d', delimiter=',')
#   np.savetxt(f'{output_path}/TIN_points.csv', p_all, fmt='%f', delimiter=',')