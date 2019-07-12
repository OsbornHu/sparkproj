package com.sparkml.mytest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.SQLTransformer

/**

SQLTransformer工具用来转换由SQL定义的陈述。目前仅支持SQL语法如"SELECT ...FROM __THIS__ ..."，其中"__THIS__"代表输入数据的基础表。选择语句指定输出中展示的字段、元素和表达式，支持Spark SQL中的所有选择语句。用户可以基于选择结果使用Spark SQL建立方程或者用户自定义函数。SQLTransformer支持语法示例如下：

1. SELECTa, a + b AS a_b FROM __THIS__

2. SELECTa, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5

3. SELECTa, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b

示例：

假设我们有如下DataFrame包含id，v1，v2列：
id | v1 | v2

----|-----|-----

0 | 1.0 | 3.0

2 | 2.0 | 5.0

使用SQLTransformer语句"SELECT *,(v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__"转换后得到输出如下：

id | v1 | v2 | v3 | v4

----|-----|-----|-----|-----

0 | 1.0| 3.0 | 4.0 | 3.0

2 | 2.0| 5.0 | 7.0 |10.0
  *
  *
  */

object SQLTransformerTest1 {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("SQLTransformer")
      .master("local[2]")
      .getOrCreate()


    val df = spark.createDataFrame(
      Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")

    val sqlTrans = new SQLTransformer().setStatement(
      "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
    sqlTrans.transform(df).show()
  }
}
