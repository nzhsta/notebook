from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 创建 Spark 会话
spark = SparkSession.builder.appName("RandomForestExample").getOrCreate()

# 假设你有一个包含特征和标签列的 DataFrame
# 这里使用一个简单的例子，你需要替换成你自己的数据
data = [("Class1", "A", 0.1, '1'),
        ("Class0", "B", 0.4, '0'),
        ("Class1", "C", 0.6, '1'),
        ("Class0", "A", 0.8, '0')]

columns = ["label", "category", "numeric_feature", "boolean_feature"]
df = spark.createDataFrame(data, columns)

# 使用 StringIndexer 将标签列转换为数值
label_indexer = StringIndexer(inputCol="label", outputCol="label_index").fit(df)
# df_indexed = label_indexer.transform(df)

# 使用 StringIndexer 将分类变量转换为数值
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(df) for col in ["category", "boolean_feature"]]
pipeline_indexers = Pipeline(stages=indexers)
# df_indexed = pipeline_indexers.fit(df_indexed).transform(df_indexed)

# 使用 OneHotEncoder 将数值化的分类变量转换为独热编码
encoder = OneHotEncoder(inputCols=["category_index", "boolean_feature_index"],
                        outputCols=["category_onehot", "boolean_feature_onehot"])
# df_encoded = encoder.fit(df_indexed).transform(df_indexed)

# 合并所有特征列
feature_columns = ["numeric_feature", "category_onehot", "boolean_feature_onehot"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
# df_assembled = assembler.transform(df_encoded)

# 划分数据集为训练集和测试集
(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

# 创建随机森林模型
rf = RandomForestClassifier(labelCol="label_index", featuresCol="features", numTrees=10)

# 创建一个 Pipeline 包含特征合并和随机森林模型
pipeline = Pipeline(stages=[label_indexer, pipeline_indexers, encoder, assembler, rf])

# 在训练集上训练模型
model = pipeline.fit(training_data)

# 在测试集上进行预测
predictions = model.transform(test_data)

# 评估模型性能
evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# 关闭 Spark 会话
spark.stop()
