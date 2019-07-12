package com.sparkml.mytest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
/**
VectorAssembler是一个转换器，它将给定的若干列合并为一列向量。它可以将原始特征和一系列通过其他转换器得到的特征合并为单一的特征向量，来训练如逻辑回归和决策树等机器学习算法。VectorAssembler可接受的输入列类型：数值型、布尔型、向量型。输入列的值将按指定顺序依次添加到一个新向量中。

示例：

假设我们有如下DataFrame包含id, hour, mobile, userFeatures以及clicked列：

id | hour | mobile| userFeatures | clicked

----|------|--------|------------------|---------

0 |18 | 1.0 | [0.0, 10.0, 0.5] | 1.0

userFeatures列中含有3个用户特征。我们想将hour, mobile以及userFeatures合并为一个新列。将VectorAssembler的输入指定为hour, mobile以及userFeatures，输出指定为features，通过转换我们将得到以下结果：

id | hour | mobile| userFeatures | clicked | features

----|------|--------|------------------|---------|-----------------------------

0 |18 | 1.0 | [0.0, 10.0, 0.5] | 1.0 | [18.0, 1.0, 0.0, 10.0, 0.5]
  *
  */

object VectorAssemblerTest1 {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("VectorAssembler")
      .master("local[2]")
      .getOrCreate()

    val dataset = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")

    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")

    val output = assembler.transform(dataset)
    println(output.select("features", "clicked").first())
  }
}
