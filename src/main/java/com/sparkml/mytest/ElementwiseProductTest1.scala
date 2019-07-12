package com.sparkml.mytest

import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
ElementwiseProduct按提供的“weight”向量，返回与输入向量元素级别的乘积。即是说，按提供的权重分别对输入数据进行缩放，得到输入向量v以及权重向量w的Hadamard积。

\left( \begin{array}{l}{v_1}\\...\\{v_n}\end{array} \right)*\left( \begin{array}{l}{w_1}\\...\\{w_n}\end{array} \right) = \left( \begin{array}{l}{v_1}{w_1}\\...\\{v_n}{w_n}\end{array} \right)
下面例子展示如何通过转换向量的值来调整向量。
  *
  */
object ElementwiseProductTest1 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ElementwiseProduct")
      .master("local[2]")
      .getOrCreate()
    // Create some vector data; also works for sparse vectors
    val dataFrame = spark.createDataFrame(Seq(
      ("a", Vectors.dense(1.0, 2.0, 3.0)),
      ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")

    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct()
      .setScalingVec(transformingVector)
      .setInputCol("vector")
      .setOutputCol("transformedVector")

    // Batch transform the vectors to create new column:
    transformer.transform(dataFrame).show()

  }
}
