from delaunay import delaunay_triangulation
import pandas as pd
import numpy as np
import time
import os


def grid_to_TIN(df_p: pd.DataFrame, max_error=0.05, output_name=None):
  start = time.time() # 実行時間計算用
  df_old = df_p.copy(deep = True)
  p_amount = len(df_p) # 点の数
  error = np.full(p_amount, np.inf) # 誤差を格納するためのリスト

  # ============================================================
  # ステップ 1: すべての点について標高誤差を計算(1回目のループ)
  #            それ以降は，削除した点に隣接する点の標高誤差を計算
  # ============================================================  
  while 1:
    for i in range(df_p.shape[0]):
      # 標高誤差がinfとなっている点のみ計算する
      if (error[i] < np.inf): continue
      df_points = df_p.drop(index = i)
      point = df_p.iloc[i][['x', 'y']].to_numpy()
      height = delaunay_triangulation(df_points, point, return_height=True)
      error[i] = abs(df_p.iloc[i]['h'] - height)

    # ============================================================
    # ステップ 2: 最小誤差が設定した最大誤差を超えたら終了
    # ============================================================
    print(error.min())
    if (error.min() > max_error): break

    # ============================================================
    # ステップ 3 ~ 4: 削除する点に隣接する点の誤差を初期化して削除
    # ============================================================
    del_id = error.argmin()
    # 削除する点に隣接(接続)する点を取得
    tri = delaunay_triangulation(df_p, p_num=del_id, return_adjacent_points=True)
    # 隣接点における誤差を初期化
    error[tri] = np.inf
    
    # 頂点情報から削除する点を削除
    df_p = df_p.drop(index = del_id)
    df_p = df_p.reset_index(drop = True)
    # 標高誤差から削除する点を削除
    error = np.delete(error, del_id)
  
  print('fin')
  print(time.time() - start, 's')

  # output用のフォルダをなければ作成
  output_path = f'./output/{output_name}/{max_error}'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  triangles = delaunay_triangulation(df_p, plot=True, output_name=f'{output_path}/TIN', return_triangles=True)
  triangles = np.array(triangles[0])

  np.savetxt(f'{output_path}/TIN_triangles.csv', triangles, fmt='%d')
  np.savetxt(f'{output_path}/TIN_points.csv', df_p[['x', 'y', 'h']].to_numpy(), fmt='%f')


if __name__ == '__main__':
  folder = './csv'
  filename = 'grid_400points.csv'
  max_error = 0.1
  output_name = filename.split('.')[0]
  df_p = pd.read_csv(f'{folder}/{filename}')
  grid_to_TIN(df_p, max_error=max_error, output_name=output_name)
