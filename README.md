# Algorithm_TIN


## TINアルゴリズム
1. TINに含まれるすべての頂点$v$について :
    
    * 一時的に頂点 $v$ を削除．
    * 削除した頂点を除いて，Delaunay三角網を構築．
    * 新たなTINを用いて，頂点 $v$ の標高誤差 $error(v)$ を計算．

    ソートされた各頂点 $v$ の誤差 $error(v)$ を 平衡二分木(balanced binary tree) $T$ に格納する．$T$ の各ノードは誤差 $error(v)$ とTINの頂点 $v$ へのポインタを保持している．  
    また頂点 $v$ には，$T$ の対応するノードへのポインタを格納する.

2. $T$ において，最小の $error(v)$ を持つノードについて検討する．もしその $error(v)$ が，設定した最大誤差よりも大きければ，ここでアルゴリズムを終了する．そうでないとき，次のステップへ進む．

3. $error(v)$ を持つ $T$ のノードを削除する．そして対応する頂点 $v$ をTIN構造体から削除する．  
頂点 $v$ の隣接する頂点を $w_1, w_2, ..., w_j$ とする．そして頂点 $w_1, w_2, ..., w_j$ を用いて，再度ドローネ三角分割を行う．

4. すべての頂点 $w_i \in \{w_1, w_2, ..., w_j\}$ について : 

    * $error(w_i)$ を保持しているノードを $T$ から削除．
    * Step1で行ったように，頂点 $w_i$ を削除した際の標高誤差 $error(w_i)$ を計算．
    * 新たに求めた $error(w_i)$ を $T$ に代入．

Step2へ続く  


#### 参考文献
> Marc van Kreveld，Jürg Nievergelt，Thomas Roos，Peter Widmayer (Eds.): Algorithmic Foundations of Geographic Information Systems，Springer，pp．47-50，1997．



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



