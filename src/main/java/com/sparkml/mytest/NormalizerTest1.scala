package com.sparkml.mytest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Normalizer

/**
Normalizer是一个转换器，它可以将多行向量输入转化为统一的形式。
参数为p（默认值：2）来指定正则化中使用的p-norm。正则化操作可以使输入数据标准化并提高后期学习算法的效果。

下面的例子展示如何读入一个libsvm格式的数据，然后将每一行转换为\[{L^2}\]以及\[{L^\infty }\]形式。
  *
  */
object NormalizerTest1 {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder()
      .appName("Normalizer")
      .master("local[2]")
      .getOrCreate()
    val data = sparkSession.read.format("libsvm").load("./data/mllib/sample_libsvm_data.txt")
    // Normalize each Vector using $L^1$ norm.
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    val l1NormData = normalizer.transform(data)
    l1NormData.show()

    // Normalize each Vector using $L^\infty$ norm.
    val lInfNormData = normalizer.transform(data, normalizer.p -> Double.PositiveInfinity)
    lInfNormData.show()
  }
}
