
本篇笔记主要是对 house price 比赛的优秀案例进行总结的笔记
- [x] 阅读特征工程笔记[stack_regression 阅读笔记](stack_regression%20阅读笔记.md)
- [x] 阅读[Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models)


	以上两篇文章都会对完成这次**回归任务**有很大的帮助



# 1. 了解数据
1. 首先了解训练集和测试集的样本数和特征数
2. 去除显而易见的无用特征('ID')


# 2. 数据预处理

## 2.1 异常值

对于目标值和特征画出散点图，可以比较清晰的**看出**是否具有比较离谱的异常值，如下图所示：
![](images/Pasted%20image%2020230430002752.png)





