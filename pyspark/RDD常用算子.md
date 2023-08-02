# 常用算子详解

## 1	RDD常用操作

### 1.1	transformers

- 作用
  可以根据一个已经存在的RDD创建一个新的RDD(RDDA —— transformer ——> RDDB)

- 例子

  - `map`
  - `filter`
  - `group by`
  - `distinct`

- 特点
  所有的transformer操作都是 **lazy** 的
  transformer不会立刻计算，只会记录(traggers)这些RDD的transformer，直到一个`action`请求返回结果到drive端，才会执行transformer，这使得Spark运行更为高效

  

### 1.2	actions

- 作用

  1. RDD经过一些计算，返回一个值到drive端
  2. 将数据写出到外部存储中

  以上两个操作都属于`action`

- 例子

  - `reduce`
  - `count`
  - `collect`



## 2	Transformer算子

### 2.1 map

- 作用
  将function函数作用到数据集的**每一个元素**上，并生成一个新的RDD



## 3	 Action算子

## 4	Spark RDD案例实战

