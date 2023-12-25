from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 创建 Spark 会话
spark = SparkSession.builder.appName("RandomForestExample").getOrCreate()

# 假设你有一个包含特征和标签列的 DataFrame
# 这里使用一个简单的例子，你需要替换成你自己的数据
data = [(1, 0.1, 0.2, 1.0),
        (0, 0.4, 0.5, 2.0),
        (1, 0.6, 0.7, 3.0),
        (0, 0.8, 0.9, 4.0)]

columns = ["label", "feature1", "feature2", "feature3"]
df = spark.createDataFrame(data, columns)
# 合并特征列，并指定一个新的输出列名
feature_columns = ["feature1", "feature2", "feature3"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="assembled_features_temp")
df = assembler.transform(df)
df = df.select(['assembled_features_temp', 'label'])
# 划分数据集为训练集和测试集
(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

# 确保删除已存在的列

# 创建随机森林模型
rf = RandomForestClassifier(labelCol="label", featuresCol="assembled_features_temp", numTrees=10)

# 创建一个 Pipeline 包含特征合并和随机森林模型
# pipeline = Pipeline(stages=[assembler, rf])

# 在训练集上训练模型
model = rf.fit(training_data)

# 在测试集上进行预测
predictions = model.transform(test_data)

# 评估模型性能
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# 关闭 Spark 会话
spark.stop()
