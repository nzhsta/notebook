# 常用算子详解

## 1	RDD常用操作

### 1.1	transformers

- 作用
  可以根据一个已经存在的RDD创建一个新的RDD(RDDA —— transformer ——> RDDB)
- 例子
  - `map`
- 特点
  所有的transformer操作都是 **lazy** 的
  transformer不会立刻计算，只会记录这些RDD的transformer，直到一个`action`请求返回结果到drive端，才会执行transformer，这使得Spark

### 1.2	actions

- 作用
  RDD经过一些计算，可以返回一个值到drive端
- 例子
  - `reduce`















## 2	Transformer算子

## 3	 Action算子

## 4	Spark RDD案例实战

