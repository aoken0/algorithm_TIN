from delaunay import delaunay_triangulation
import pandas as pd
import numpy as np
import time
import os
import copy


# def grid_to_TIN(df_p: pd.DataFrame, max_error=0.05, output_name=None):
#   start = time.time() # 実行時間計算用
#   df_old = df_p.copy(deep = True)
#   p_amount = len(df_p) # 点の数
#   error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト

#   # ============================================================
#   # ステップ 1: すべての点について標高誤差を計算(1回目のループ)
#   #            それ以降は，削除した点に隣接する点の標高誤差を計算
#   # ============================================================  
#   while 1:
#     for i in range(df_p.shape[0]):
#       # 標高誤差がinfとなっている点のみ計算する
#       if (error[i] < np.inf): continue
#       df_points = df_p.drop(index = i)
#       point = df_p.iloc[i][['x', 'y']].to_numpy()
#       height = delaunay_triangulation(df_points, point, return_height=True)
#       error[i] = abs(df_p.iloc[i]['h'] - height)

#     # ============================================================
#     # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
#     # ============================================================
#     print(error.min())
#     if (error.min() > max_error): break

#     # ============================================================
#     # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を初期化して削除
#     # ============================================================
#     del_id = error.argmin()
#     # 削除する点に隣接(接続)する点を取得
#     tri = delaunay_triangulation(df_p, p_num=del_id, return_adjacent_points=True)
#     # 隣接点における誤差を初期化
#     error[tri] = np.inf
    
#     # 頂点情報から削除する点を削除
#     df_p = df_p.drop(index = del_id)
#     df_p = df_p.reset_index(drop = True)
#     # 標高誤差から削除する点を削除
#     error = np.delete(error, del_id)
  
#   print('fin')
#   print(time.time() - start, 's')

#   # output用のフォルダをなければ作成
#   output_path = f'./output/{output_name}/{max_error}'
#   if not os.path.exists(output_path):
#     os.makedirs(output_path)
  
#   triangles = delaunay_triangulation(df_p, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
#   triangles = np.array(triangles[0])

#   np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d')
#   np.savetxt(f'{output_path}/TIN_points.csv', df_p[['x', 'y', 'h']].to_numpy(), fmt='%f')


# def grid_to_TIN_1(df_p: pd.DataFrame, max_error=0.05, output_name=None):
#   start = time.time() # 実行時間計算用
#   p_amount = len(df_p) # 点の数
#   error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
#   points = df_p[['x', 'y', 'h']].to_numpy()

#   # ============================================================
#   # ステップ 1: すべての点について標高誤差を計算(1回目のループ)
#   #            それ以降は，削除した点に隣接する点の標高誤差を計算
#   # ============================================================  
#   while 1:
#     for i in range(points.shape[0]):
#       # 標高誤差がinfとなっている点のみ計算する
#       if (error[i] < np.inf): continue
#       point = points[i, :2]
#       tmp_points = np.delete(points, i, axis=0)
#       height = delaunay_triangulation(tmp_points, point, return_height=True)
#       error[i] = abs(points[:, 2][i] - height)

#     # ============================================================
#     # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
#     # ============================================================
#     print(error.min())
#     if (error.min() > max_error): break

#     # ============================================================
#     # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を初期化して削除
#     # ============================================================
#     del_id = error.argmin()
#     # 削除する点に隣接(接続)する点を取得
#     tri = delaunay_triangulation(points, p_num=del_id, return_adjacent_points=True)
#     # 隣接点における誤差を初期化
#     error[tri] = np.inf
    
#     # 頂点情報から削除する点を削除
#     points = np.delete(points, del_id, axis=0)
#     # 標高誤差から削除する点を削除
#     error = np.delete(error, del_id)
  
#   print('fin')
#   print(time.time() - start, 's')

#   # output用のフォルダをなければ作成
#   output_path = f'./output/{output_name}/{max_error}_1'
#   if not os.path.exists(output_path):
#     os.makedirs(output_path)
  
#   triangles = delaunay_triangulation(points, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
#   triangles = np.array(triangles[0])

#   np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d')
#   np.savetxt(f'{output_path}/TIN_points.csv', points, fmt='%f')


# def grid_to_TIN_2(df_p: pd.DataFrame, row=50, col=50, x_step=1, y_step=1, max_error=0.05, output_name=None):
#   start = time.time() # 実行時間計算用
#   p_amount = len(df_p) # 点の数
#   error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
#   points = df_p[['x', 'y', 'h']].to_numpy() # 頂点情報をndarrayに変換
#   points_reshape = np.reshape(points, (row, col, 3)) # 3次元配列に変換

#   # ============================================================
#   # ステップ 1: すべての点について標高誤差を計算
#   # ============================================================  
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
    
#   while 1:
#     for i in range(points.shape[0]):
#       # 標高誤差がinfとなっている点のみ計算する
#       if (error[i] < np.inf): continue
#       point = points[i, :2]
#       tmp_points = np.delete(points, i, axis=0)
#       height = delaunay_triangulation(tmp_points, point, return_height=True)
#       error[i] = abs(points[:, 2][i] - height)

#     # ============================================================
#     # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
#     # ============================================================
#     print(error.min())
#     if (error.min() > max_error): break

#     # ============================================================
#     # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を初期化して削除
#     # ============================================================
#     del_id = error.argmin()
#     # 削除する点に隣接(接続)する点を取得
#     tri = delaunay_triangulation(points, p_num=del_id, return_adjacent_points=True)
#     # 隣接点における誤差を初期化
#     error[tri] = np.inf
    
#     # 頂点情報から削除する点を削除
#     points = np.delete(points, del_id, axis=0)
#     # 標高誤差から削除する点を削除
#     error = np.delete(error, del_id)
  
#   print('fin')
#   print(time.time() - start, 's')

#   # output用のフォルダをなければ作成
#   output_path = f'./output/{output_name}/{max_error}_2'
#   if not os.path.exists(output_path):
#     os.makedirs(output_path)
  
#   triangles = delaunay_triangulation(points, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
#   triangles = np.array(triangles[0])

#   np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d')
#   np.savetxt(f'{output_path}/TIN_points.csv', points, fmt='%f')


def grid_to_TIN_3(df_p: pd.DataFrame, row=50, col=50, x_step=1, y_step=1, max_error=0.05, output_name=None):
  start = time.time() # 実行時間計算用
  p_amount = len(df_p) # 点の数
  error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト
  points = df_p[['x', 'y', 'h']].to_numpy() # 頂点情報をndarrayに変換
  points_reshape = np.reshape(points, (row, col, 3)) # 3次元配列に変換

  # ============================================================
  # ステップ 1: すべての点について標高誤差を計算
  # ============================================================  
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
  output_path = f'./output/{output_name}/{max_error}_3'
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
  filename = 'grid_22500points_shift.csv'
  [row, col] = [150, 150]
  # filename = 'grid_10000points_shift.csv'
  # [row, col] = [100, 100]
  # filename = 'grid_2500points_shift.csv'
  # filename = 'grid_400points_shift.csv'
  # [row, col] = [20, 20]
  # filename = 'grid_400points.csv'
  # [row, col] = [20, 20]
  # filename = 'IzuOshima10m.csv'
  max_error = 0.1
  output_name = filename.split('.')[0]
  df_p = pd.read_csv(f'{folder}/{filename}')
  # grid_to_TIN(df_p, max_error=max_error, output_name=output_name)
  # grid_to_TIN_1(df_p, max_error=max_error, output_name=output_name)
  # grid_to_TIN_2(df_p, row=row, col=col, x_step=1, y_step=1.1, max_error=max_error, output_name=output_name)
  grid_to_TIN_3(df_p, row=row, col=col, x_step=1, y_step=1.1, max_error=max_error, output_name=output_name)
