
## 1 RDD 的特性
- 特性
	  1. RDD 是不可变的
	  2. 一系列 partition 构成
	  3. 一个函数可以对所有 partition 进行计算
	     对一个 rdd 执行一个函数，实际上是对 RDD 的所有分区数据执行某种操作
	     $y=f(x)$
	     $rdd.map(\_+1)$
	  4. RDD 是一个依赖<font color="#2DC26B">其他 RDD </font>的列表
	     RDD\==> RDD 1\==> RDD 2\==> RDD 3
	     RDD 之间存在依赖关系
	     例子：RDDA（5 partition）\==>RDDB (partition)