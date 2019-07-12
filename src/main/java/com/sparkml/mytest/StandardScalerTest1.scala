package com.sparkml.mytest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StandardScaler

/**
StandardScaler处理Vector数据，标准化每个特征使得其有统一的标准差以及（或者）均值为零。它需要如下参数：

1. withStd：默认值为真，使用统一标准差方式。

2. withMean：默认为假。此种方法将产出一个稠密输出，所以不适用于稀疏输入。

StandardScaler是一个Estimator，它可以fit数据集产生一个StandardScalerModel，用来计算汇总统计。然后产生的模可以用来转换向量至统一的标准差以及（或者）零均值特征。

注意如果特征的标准差为零，则该特征在向量中返回的默认值为0.0。

下面的示例展示如果读入一个libsvm形式的数据以及返回有统一标准差的标准化特征。
  *
  */
object StandardScalerTest1 {

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder()
      .appName("StandardScaler")
      .master("local[2]")
      .getOrCreate()
    val data = sparkSession.read.format("libsvm").load("./data/mllib/sample_libsvm_data.txt")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(data)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(data)
    scaledData.show()

  }
}
