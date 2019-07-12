package com.sparkml.mytest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.MaxAbsScaler

/**
MaxAbsScaler使用每个特征的最大值的绝对值将输入向量的特征值转换到[-1,1]之间。因为它不会转移／集中数据，所以不会破坏数据的稀疏性。

下面的示例展示如果读入一个libsvm形式的数据以及调整其特征值到[-1,1]之间。
  *
  */
object MaxAbsScalerTest1 {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder()
      .appName("MaxAbsScaler")
      .master("local[2]")
      .getOrCreate()
    val data = sparkSession.read.format("libsvm").load("./data/mllib/sample_libsvm_data.txt")
    val scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MaxAbsScalerModel
    val scalerModel = scaler.fit(data)

    // rescale each feature to range [-1, 1]
    val scaledData = scalerModel.transform(data)
    scaledData.show()
  }
}
