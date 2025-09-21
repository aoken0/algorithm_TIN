from matplotlib import pyplot as plt
from delaunay import delaunay_triangulation, plot_network
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

  # s0 = s1 = s2 = s3 = s4 = s5 = 0

  # ============================================================
  # ステップ 1: すべての点について標高誤差を計算
  # ============================================================
  # tt = time.time()
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
  # s0 += time.time() - tt

  # ドローネ分割を行い、すべての三角形を取得
  triangle_all: np.ndarray = delaunay_triangulation(p_all, return_triangles=True)[0]
  # 三角形の情報から隣接リストを作成
  adjacency = build_adjacency_list(triangle_all, p_all.shape[0])

  # tt = time.time()
  # s1 += time.time() - tt

  while 1:
    # ============================================================
    # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
    # ============================================================
    if (error.min() > max_error): break
    del_amount += 1
    if del_amount % 1000 == 0:
      print(del_amount)
    # 削除点の配列番号
    del_id = error.argmin()

    # ============================================================
    # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を再計算
    # ============================================================
    # tt = time.time()
    # 削除点から3つ先までの点を取得する
    neighbor_points = get_neighbor_points(del_id, adjacency, depth=[1,2,3])
    # print(np.sort(neighbor_point_indexes[0]),np.sort(neighbor_point_indexes[1]),np.sort(neighbor_point_indexes[2]))
    p_adjacent = neighbor_points[0]
    p_neighbor_depth2 = neighbor_points[1]
    p_neighbor = neighbor_points[2]
    # neighbor_point_indexes_old = get_neighbor_points_old(del_id, triangle_all, depth=3)
    # print(np.sort(neighbor_point_indexes_old[0]),np.sort(neighbor_point_indexes_old[1]),np.sort(neighbor_point_indexes_old[2]))
    # s2 += time.time() - tt

    # tt = time.time()
    # 削除点に隣接する点全てにおいて削除して誤差計算する
    for i in p_adjacent:
      p_del = p_all[i, :2] # 隣接点の抽出(削除して誤差を求める対象点)
      p_tmp = p_neighbor[p_neighbor != i]
      p_tmp_neighbor = p_all[p_tmp]
      height = delaunay_triangulation(p_tmp_neighbor, p_del, return_height=True)
      error[i] = abs(p_all[:, 2][i] - height)
    # s3 += time.time() - tt

    # ============================================================
    # 削除点を除いたdelaunayネットワークを再構築
    # ============================================================
    # tt = time.time()
    # 深さ2の隣接点までを用いて計算する
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
    # 隣接リストを更新
    adjacency = update_adjacency_list(adjacency, del_id, filtered_new_triangle)
    # s4 += time.time() - tt

    # tt = time.time()
    # 削除点に関する要素を無限大に(あとではじく)
    p_all[del_id] = [np.inf, np.inf, np.inf]
    error[del_id] = np.inf
    # s3 += time.time() - s
    # s5 += time.time() - tt
    return

  print('fin')
  print(time.time() - start, 's')
  # print(s0, s1, s2, s3, s4, s5)

  # ============================================================
  # 出力等
  # ============================================================
  # output用のフォルダをなければ作成
  output_path = f'./output/{output_name}/{max_error}'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  p = p_all[~np.all(np.isin(p_all, np.inf), axis=1)]
  triangles = delaunay_triangulation(p, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
  triangles = np.array(triangles[0])
  # plot_network(p_all[:, :2], p_all[:, 2], triangle_all, output_name=f'{output_path}/TIN')
  np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d', delimiter=',')
  np.savetxt(f'{output_path}/TIN_points.csv', p, fmt='%f', delimiter=',') 


def build_adjacency_list(triangles: np.ndarray, p_amount: int):
  adjacency = {i: set() for i in range(p_amount)}
  for tri in triangles:
    adjacency[tri[0]].update([tri[1], tri[2]])
    adjacency[tri[1]].update([tri[2], tri[0]])
    adjacency[tri[2]].update([tri[0], tri[1]])
  return adjacency

def update_adjacency_list(adjacency: dict[int, set], p_del: int, new_triangles: np.ndarray):
  adjacency = remove_point_from_adjacency(adjacency, p_del)
  adjacency = add_point_to_adjacency(adjacency, new_triangles)
  return adjacency

def remove_point_from_adjacency(adjacency: dict[int, set], p_del):
  # 削除点の隣接点を取得
  neighbors: dict[int, set] = adjacency[p_del]
  # 隣接リストから削除点を削除
  adjacency.pop(p_del, None)
  # setから削除
  for neighbor in neighbors:
    adjacency[neighbor].discard(p_del)
  
  return adjacency

def add_point_to_adjacency(adjacency: dict[int, set], new_triangles: np.ndarray):
  for tri in new_triangles:
    adjacency[tri[0]].update([tri[1], tri[2]])
    adjacency[tri[1]].update([tri[2], tri[0]])
    adjacency[tri[2]].update([tri[0], tri[1]])
  return adjacency

def get_neighbor_points(p_reference: int, adjacency: dict[int, set], depth: list[int]):
  # depthを昇順に並び替え
  depth.sort()
  d_cnt = 0
  p_references = set([p_reference])
  neighbors = set()
  return_list = []
  for i in range(max(depth)):
    for j in p_references:
      neighbors.update(adjacency[j])
    p_references = neighbors - p_references
    if depth[d_cnt] == i + 1:
      d_cnt += 1
      return_list.append(np.array(list(neighbors - {p_reference})))
  
  return return_list

if __name__ == '__main__':
  folder = './csv'
  # filename = 'test/grid_1000000points_shift.csv'
  # [row, col] = [1000, 1000]
  # filename = 'test/grid_22500points_shift.csv'
  # [row, col] = [150, 150]
  # filename = 'test/grid_10000points_shift.csv'
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
