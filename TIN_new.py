import numpy as np
from scipy.spatial import Delaunay
import os
import time
import pandas as pd

def grid_to_TIN_optimized(df_p: pd.DataFrame, max_error=0.05, output_name=None):
    start = time.time()
    
    # 初期化
    points = df_p[['x', 'y']].to_numpy()
    heights = df_p['h'].to_numpy()
    delaunay = Delaunay(points)

    # エラー初期化
    error = np.full(len(points), np.inf)
    active_points = np.arange(len(points))
    
    while True:
        # ステップ 1: エラー計算（未計算のみ）
        for i in range(len(points)):
            if error[i] < np.inf:
                continue
            simplex_index = delaunay.find_simplex(points[i])
            if simplex_index == -1:
                continue
            simplex = delaunay.simplices[simplex_index]
            triangle_points = points[simplex]
            triangle_heights = heights[simplex]
            error[i] = compute_height_error(triangle_points, triangle_heights, points[i], heights[i])
        
        # ステップ 2: 最小誤差が最大誤差を超えたら終了
        min_error_index = np.argmin(error)
        if error[min_error_index] > max_error:
            break
        
        # ステップ 3: 点を削除し隣接点のエラーを初期化
        neighbors = delaunay.vertex_neighbor_vertices
        start_index, end_index = neighbors[0][min_error_index], neighbors[0][min_error_index + 1]
        adjacent_points = neighbors[1][start_index:end_index]
        
        # 隣接点のエラーを初期化
        error[adjacent_points] = np.inf
        error[min_error_index] = np.inf
        
        # 削除
        active_points = np.delete(active_points, min_error_index)
        points = np.delete(points, min_error_index, axis=0)
        heights = np.delete(heights, min_error_index, axis=0)
        delaunay = Delaunay(points)
    
    print('Finished in:', time.time() - start, 'seconds')
    
    # 結果出力
    output_path = f'./output/{output_name}/{max_error}'
    os.makedirs(output_path, exist_ok=True)
    np.savetxt(f'{output_path}/TIN_points.csv', np.hstack((points, heights.reshape(-1, 1))), fmt='%f')
    np.savetxt(f'{output_path}/TIN_triangles.csv', delaunay.simplices, fmt='%d')

def compute_height_error(triangle_points, triangle_heights, point, true_height):
    # 三角形の全体の面積を計算
    vectors = [
        triangle_points[1] - triangle_points[0],
        triangle_points[2] - triangle_points[0]
    ]
    full_area = 0.5 * abs(np.cross(vectors[0], vectors[1]))

    # 内包された点と他2点からなる三角形の面積を3通り計算
    a0 = compute_area(point, triangle_points[1], triangle_points[2]) / full_area
    a1 = compute_area(point, triangle_points[2], triangle_points[0]) / full_area
    a2 = compute_area(point, triangle_points[0], triangle_points[1]) / full_area

    # 標高の補間値を計算
    interpolated_height = a0 * triangle_heights[0] + a1 * triangle_heights[1] + a2 * triangle_heights[2]
    return abs(interpolated_height - true_height)

def compute_area(p1, p2, p3):
    # 三角形の面積を計算
    vectors = np.array([p2 - p1, p3 - p1])
    return 0.5 * abs(np.cross(vectors[0], vectors[1]))

if __name__ == '__main__':
  folder = './csv'
  filename = 'grid_400points.csv'
  # filename = 'IzuOshima10m.csv'
  max_error = 0.1
  output_name = filename.split('.')[0]
  df_p = pd.read_csv(f'{folder}/{filename}')
  grid_to_TIN_optimized(df_p, max_error=max_error, output_name=output_name)
