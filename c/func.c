#include <stdio.h>
#include <math.h>

void get_vector(double *vector, const double *p1, const double *p2) {
  vector[0] = p2[0] - p1[0];
  vector[1] = p2[1] - p1[1];
}

double get_triangle_area(double *vector0, double *vector1) {
  return 0.5 * fabs(vector0[0]*vector1[1]-vector0[1]*vector1[0]);
}

double get_height(const double triangle_points[][3], double *p_target) {
  double vector0[2] = {0.0, 0.0}; // 計算用
  double vector1[2] = {0.0, 0.0};
  
  // 全体の三角形の面積
  get_vector(vector0, triangle_points[0], triangle_points[1]);
  get_vector(vector1, triangle_points[0], triangle_points[2]);
  double full_area = get_triangle_area(vector0, vector1);
  
  // 内包された点と他2点から成る三角形の面積を3通り求める
  // # 点0, 1 -> a2
  get_vector(vector0, triangle_points[0], triangle_points[1]);
  get_vector(vector1, triangle_points[0], p_target);
  double a2 = get_triangle_area(vector0, vector1) / full_area;
  // 点1, 2 -> a0
  get_vector(vector0, triangle_points[1], triangle_points[2]);
  get_vector(vector1, triangle_points[1], p_target);
  double a0 = get_triangle_area(vector0, vector1) / full_area;
  // 点2, 0 -> a1
  get_vector(vector0, triangle_points[2], triangle_points[0]);
  get_vector(vector1, triangle_points[2], p_target);
  double a1 = get_triangle_area(vector0, vector1) / full_area;

  return a0 * triangle_points[0][2] + a1 * triangle_points[1][2] + a2 * triangle_points[2][2];
}