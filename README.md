## ドローネ三角分割(Delaunay Triangulation)

アルゴリズムに関する情報は，[Algorithm_Delaunay](https://github.com/aoken0/Algorithm_Delaunay) を参照．


## 重心座標系(Barycentric coordinate system)

ある三角形に内包されている点について、標高を線形補間で求める。

### 数式等
任意の点を $P$ ，点 $P$ を内包する三角形の頂点をそれぞれ $A, B, C$ とする．またそれぞれの点の標高(重み)は $P_h, A_h, B_h, C_h$ とする．  
重心座標系を用いると任意の点 $P$ の座標・標高は以下のように表せる．

```math
\begin{align*}
P &= \lambda_1 A + \lambda_2 B + \lambda_2 C \\
P_h &= \lambda_1 A_h + \lambda_2 B_h + \lambda_2 C_h \\
\end{align*}
```

&emsp;

そして $\lambda_1, \lambda_2, \lambda_3$ は以下のように求められる．  
面積を $area()$ として， 

```math
\begin{align*}
\lambda_1 = area(PBC) / area(ABC) \\
\lambda_2 = area(APC) / area(ABC) \\
\lambda_3 = area(ABP) / area(ABC) \\
\end{align*}
```

となる．また

```math
\begin{align*}
1 = \lambda_1 + \lambda_2 + \lambda_3
\end{align*}
```

である．

ここで面積はベクトルの外積を用いて求めた．  
例)

```math
\begin{align*}
area(ABC) = \frac{1}{2} AB \times AC
\end{align*}
```

> 参考  
> * https://en.wikipedia.org/wiki/Barycentric_coordinate_system



