
本篇笔记主要是对 house price 比赛的优秀案例进行总结的笔记
- [x] 阅读特征工程笔记[stack_regression 阅读笔记](stack_regression%20阅读笔记.md)
- [x] 阅读[Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models)

以上两篇文章都会对完成这次**回归任务**有很大的帮助



# 1. 了解数据

1. 首先**了解训**练集和测**试集**的**样本数和特征数
2. 去除显而易见的无用**特**征('ID')



# 2. 数据处理

## 2.1 异常值

对于目标值和特征画出散点图，可以比较清晰的看出是否具有比较离谱的异常值，如下图所示：
![](images/Pasted%20image%2020230503210448.png)
上图可以看到，在坐标轴的右边，有两个离群点异常值（特别**极端**的异常），可以讲这两个异常样本去除。
一般情况下，直接这样去除异常样本是很不安全的，但是对于这种特别**极端**的值可以放心的去除。

- <u>可能还有其他离群点在训练数据中。但是，如果在测试数据中也有离群点，则如果将它们全部删除，则可能会对我们的模型产生不良影响。因此，我们不会删除所有的离群点，而是在建模过程中，会使一些模型对它们具有鲁棒性。</u>

## 2.2 目标变量

针对目标变量做一些分析
1. **偏度、峰度**
	利用``from scipy import stats``方法可以得到某个变量的统计分布情况
	``(mu, sigma) = norm.fit(train['SalePrice'])
2. **分布图**
	利用``sns.displot(var)``方法可以得到某个变量var的**正态分布直方图以及核密度估计**
	```python
	sns.distplot(train['SalePrice'] , fit=norm);
	# fit=norm 可以得到一条正态分布曲线
```
3. **QQ图**（检验样本数据概率分布，默认检验变量的正态分布）
	q-q 图是通过比较数据和正态分布的**分位数**是否相等来判断数据是不是符合正态分布
	```python
	stats.probplot(train['SalePrice'], plot=plt)
```
	![](images/Pasted%20image%2020230505113457.png)
			1. 可以