package com.sparkml.mytest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.QuantileDiscretizer

/**
QuantileDiscretizer讲连续型特征转换为分级类别特征。分级的数量由numBuckets参数决定。
分级的范围有渐进算法决定。渐进的精度由relativeError参数决定。当relativeError设置为0时，将会计算精确的分位点（计算代价较高）。
分级的上下边界为负无穷到正无穷，覆盖所有的实数值。

示例：
假设我们有如下DataFrame包含id, hour：

id | hour

----|------

0 |18.0

----|------

1 |19.0

----|------

2 | 8.0

----|------

3 | 5.0

----|------

4 | 2.2

hour是一个Double类型的连续特征，将参数numBuckets设置为3，我们可以将hour转换为如下分级特征。

id | hour | result

----|------|------

0 |18.0 | 2.0

----|------|------

1 |19.0 | 2.0

----|------|------

2 |8.0 | 1.0

----|------|------

3 |5.0 | 1.0

----|------|------

4 |2.2 | 0.0
  *
  */
object QuantileDiscretizerTest1 {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("QuantileDiscretizer")
      .master("local[2]")
      .getOrCreate()

    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
    var df = spark.createDataFrame(data).toDF("id", "hour")

    val discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3)

    val result = discretizer.fit(df).transform(df)
    result.show()
  }
}
