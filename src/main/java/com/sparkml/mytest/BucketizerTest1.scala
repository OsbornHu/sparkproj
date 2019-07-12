package com.sparkml.mytest

import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.SparkSession
/**
Bucketizer将一列连续的特征转换为特征区间，区间由用户指定。参数如下：

1. splits：分裂数为n+1时，将产生n个区间。除了最后一个区间外，每个区间范围［x,y］由分裂的x，y决定。分裂必须是严格递增的。在分裂指定外的值将被归为错误。两个分裂的例子为Array(Double.NegativeInfinity,0.0, 1.0, Double.PositiveInfinity)以及Array(0.0, 1.0, 2.0)。

注意，当不确定分裂的上下边界时，应当添加Double.NegativeInfinity和Double.PositiveInfinity以免越界。

下面将展示Bucketizer的使用方法。
  *
  */
object BucketizerTest1 {
  def main(args: Array[String]): Unit = {
    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
    val spark = SparkSession.builder()
      .appName("Bucketizer")
      .master("local[2]")
      .getOrCreate()
    val data = Array(-0.5, -0.3, 0.0, 0.2)
    val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketedFeatures")
      .setSplits(splits)

    // Transform original data into its bucket index.
    val bucketedData = bucketizer.transform(dataFrame)
    bucketedData.show()
  }
}
