1. fff
	1. fff
		1. 方法
		   ```python
		   class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
			    def __init__(self, base_models, meta_model, n_folds=5):
			        self.base_models = base_models
			        self.meta_model = meta_model
			        self.n_folds = n_folds
   
			    # We again fit the data on clones of the original models
			    def fit(self, X, y):
			        self.base_models_ = [list() for x in self.base_models]
			        self.meta_model_ = clone(self.meta_model)
			        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
			        # Train cloned base models then create out-of-fold predictions
			        # that are needed to train the cloned meta-model
			        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
			        for i, model in enumerate(self.base_models):
			            for train_index, holdout_index in kfold.split(X, y):
			                instance = clone(model)
			                self.base_models_[i].append(instance)
			                instance.fit(X[train_index], y[train_index])
			                y_pred = instance.predict(X[holdout_index])
			                out_of_fold_predictions[holdout_index, i] = y_pred
			                
			        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
			        self.meta_model_.fit(out_of_fold_predictions, y)
			        return self
			   
			    #Do the predictions of all base models on the test data and use the averaged predictions as 
			    #meta-features for the final prediction which is done by the meta-model
			    def predict(self, X):
			        meta_features = np.column_stack([
			            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
			            for base_models in self.base_models_ ])
			        return self.meta_model_.predict(meta_features)
			```

是

|  李宪慧  |   时间    | 事由   | 收入   | 支出  |
|:--------:|:---------:| ------ | ------ | ----- |
|   张莛   | 2022/7/31 | 结婚   |        | 1,000 |
|  张艺林  | 2022/10/3 | 结婚   |        | 200   |
| 未来爸妈 | 2022/10/1 | 见面礼 | 11,000 |       |
| 志恒大姑 | 2022/10/1 | 见面礼 | 400    |       |

