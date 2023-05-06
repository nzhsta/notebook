
本篇笔记主要是对 house price 比赛的优秀案例进行总结的笔记
- [ ] 阅读特征工程笔记[stack_regression 阅读笔记](stack_regression%20阅读笔记.md)
- [ ] 阅读[Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models)

以上两篇文章都会对完成这次**回归任务**有很大的帮助



## 1. 了解数据

1. 首先**了解训**练集和测**试集**的**样本数和特征数
2. 去除显而易见的无用**特**征('ID')



## 2. 数据处理

### 2.1 异常值

对于目标值和特征画出散点图，可以比较清晰的看出是否具有比较离谱的异常值，如下图所示：
![](images/Pasted%20image%2020230503210448.png)
上图可以看到，在坐标轴的右边，有两个离群点异常值（特别**极端**的异常)，可以讲这两个异常样本去除。
一般情况下，直接这样去除异常样本是很不安全的，但是对于这种特别**极端**的值可以放心的去除。

- <u>可能还有其他离群点在训练数据中。但是，如果在测试数据中也有离群点，则如果将它们全部删除，则可能会对我们的模型产生不良影响。因此，我们不会删除所有的离群点，而是在建模过程中，会使一些模型对它们具有鲁棒性。</u>

### 2.2 目标变量

针对目标变量做一些分析
1. **偏度、峰度**
	利用``from scipy import stats``方法可以得到某个变量的统计分布情况
	``(mu, sigma) = norm.fit(train['SalePrice'])
	利用``sns.displot(var)``方法可以得到某个变量var的**正态分布直方图以及核密度估计**
	```python
	sns.distplot(train['SalePrice'] , fit=norm);
	# fit=norm 可以得到一条正态分布曲线
```
	![500](images/Pasted%20image%2020230505113642.png)
2. **QQ图**（检验样本数据概率分布，默认检验变量的正态分布）
	q-q 图是通过比较数据和正态分布的**分位数**是否相等来判断数据是不是符合正态分布
	```python
	stats.probplot(train['SalePrice'], plot=plt)
```
	![500](images/Pasted%20image%2020230505113659.png)
	
	可以由上面两个图看到，目标变量的分布是属于**右偏态**的，但是<u>一般的线性模型都要求变量符合正态分布</u>，所以我们需要使得目标变量更加符合**正态分布**。

### 2.3 对数变换
`` np.log1p(train["SalePrice"])``利用numpy进行对数变换
```python
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

```
![](images/Pasted%20image%2020230505171504.png)
可以看到**训练数据的目标变量**经过对数变换之后，更加符合正态分布。

## 3. 特征工程
将训练数据和测试数据都concat在一起,但是去除目标变量。
```python
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
```
### 3.1 缺失值
统计出每个变量的缺失值比例，并画图
![](images/Pasted%20image%2020230505172429.png)
### 3.2 特征相似
通过sns的热力图，就可以看到变量之间的相互关系（corr相关系数）
```python
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
```
![](images/Pasted%20image%2020230505172914.png)
### 3.3 填充缺失值
通过按顺序处理缺失值的特征来填充它们（根据每个特征的特性：均值、定值、中位数、众数、None等）
注意这里不区分训练数据、测试数据

### 3.3 其他特征工程
1. **将某些数值变量转变成类别变量**
   ``all_data['MSSubClass'].apply(str)
   
2. **对一些可能包含信息的分类变量进行标签编码，以反映它们的顺序集合**
   
3. **增加特征**
   将某些特征组合（加减乘除）可以得到一个新的特征
   ```python
   from sklearn.preprocessing import LabelEncoder
	cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
	        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
	        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
	        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
	        'YrSold', 'MoSold')
	 # process columns, apply LabelEncoder to categorical features
	for c in cols:
	    lbl = LabelEncoder() 
	    lbl.fit(list(all_data[c].values)) 
	    all_data[c] = lbl.transform(list(all_data[c].values))
	
	# shape        
	print('Shape all_data: {}'.format(all_data.shape))
   
   ```

4. **解决数据倾斜**
   我们用**scipy函数boxcox1p**来计算Box-Cox转换，目标是找到一个简单的转换方式使数据规范化。
   - 首先得到哪些变量具有偏度
        ```python
		numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
		# Check the skew of all numerical features
		skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
		print("\nSkew in numerical features: \n")
		skewness = pd.DataFrame({'Skew' :skewed_feats})
		skewness.head(10)
		   ```
   - 按序对特征进行对数变换
     ```python
	    skewness = skewness[abs(skewness) > 0.75]
		print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
		
		from scipy.special import boxcox1p
		skewed_features = skewness.index
		lam = 0.15
		for feat in skewed_features:
		    #all_data[feat] += 1
		    all_data[feat] = boxcox1p(all_data[feat], lam)
	    
			#all_data[skewed_features] = np.log1p(all_data[skewed_features])
     
     ```
     
   
## 4. 建模
### 4.1 导入相关的库
```python
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
```
### 4.2 定义交叉验证策略
```python
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
	rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
	# 注意这里为负的均方误差，所以要取负号使其为正
    return(rmse)
```
### 4.3 基模型
1. **Lasso 回归**
   lasso回归对于**异常值**特别敏感，如果希望它更加**健壮**，那么可以在pipeline中使用sklearn的**Robustscaler**方法
   ``lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))``
   - 关于[make_pipeline](https://blog.csdn.net/elma_tww/article/details/88427695)
   - 关于[RobustScaler](https://scikit-learn.org.cn/view/751.html)

2. **弹性网络回归**
   同样需要对于异常值进行处理
   ``ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))``
   
3. **核-岭回归**(Kernel Ridge Regression)
   为什么需要引入**kernel**呢？其实，将kernel trick应用到distance-based的方法中是很直接的，<u>因为kernel函数本身就是一个distance-based的函数</u>。可能我们会发现，<font color=#F36208>基于distance的方法，都会有对应的一个kernel版本的扩展</font>。
   此外，从实际应用来看， 因为数据可能是**非线性**的，<u>单纯地假设真实数据服从线性关系，并用线性模型来回归真实的非线性数据，效果想必不会好</u>。所以，引入kernel还能有一个好处，就是：引入kernel的RR，也就是KRR，能够**处理非线性数据**，即，将数据映射到某一个核空间，使得数据在这个核空间上**线性可分**。
   ``KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)``
   
 4. **Gradient Boosting Regression**
    [什么是huber损失？](https://zhuanlan.zhihu.com/p/358103958)
    使用huber损失会使得模型对于异常值更加的健壮

5. **XGBoost**
   ```python
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
```

6. **LightGBM**
   ```python
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
```

### 4.4 基础模型的表现
在交叉验证的策略下查看每个基础模型的表现状况
```python
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

### 4.5 stacking model
```python
d

```
