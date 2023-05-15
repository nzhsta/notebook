1. fff
	1. fff
		1. 方法
		   ```python
		    # Finding numeric features
		   
		   numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		   numeric = []
		   for i in train.columns:
		       if train[i].dtype in numeric_dtypes:
		           if i in ['TotalSF', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'hasgarage', 'hasbsmt', 'hasfireplace']:
		               pass
		           else:
		               numeric.append(i)
		           # visualising some more outliers in the data values
		   fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))
		   plt.subplots_adjust(right=2)
		   plt.subplots_adjust(top=2)
		   sns.color_palette("husl", 8)
		   for i, feature in enumerate(list(train[numeric]), 1):
		       if (feature == 'MiscVal'):
		           break
		       plt.subplot(len(list(numeric)), 3, i)
		       sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)
		       plt.xlabel('{}'.format(feature), size=15, labelpad=12.5)
		   		plt.ylabel('SalePrice', size=15, labelpad=12.5)
		   		for j in range(2):
		         plt.tick_params(axis='x', labelsize=12)
		         plt.tick_params(axis='y', labelsize=12)
		   
		   			plt.legend(loc='best', prop={'size': 10})
		   plt.show()
```

| 李宪慧  | 时间        | 事由  |  支出     |  收入      |
|------|-----------|-----|---------|----------|
| 张莛   | 2022/7/31 | 结婚  |  1,000  |          |
| 张艺林  | 2022/10/3 | 结婚  |  200    |          |
| 未来爸妈 | 2022/10/1 | 见面礼 |         |  11,000  |
| 志恒大姑 | 2022/10/1 | 见面礼 |         |  400     |

