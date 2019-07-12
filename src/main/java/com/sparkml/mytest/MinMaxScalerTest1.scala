package com.sparkml.mytest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.MinMaxScaler

/**
MinMaxScaler通过重新调节大小将Vector形式的列转换到指定的范围内，通常为[0,1]，它的参数有：

1. min：默认为0.0，为转换后所有特征的下边界。

2. max：默认为1.0，为转换后所有特征的下边界。

MinMaxScaler计算数据集的汇总统计量，并产生一个MinMaxScalerModel。该模型可以将独立的特征的值转换到指定的范围内。

对于特征E来说，调整后的特征值如下：

\[{\mathop{\rm Re}\nolimits} scaled({e_i}) = \frac{{{e_i} - {E_{\min }}}}{{{E_{\max }} - {E_{\min }}}}*(\max  - \min ) + \min \]
如果{E_{\max }} =  &  = {E_{\min }}，则{\mathop{\rm Re}\nolimits} scaled = 0.5*(\max  - \min )。

注意因为零值转换后可能变为非零值，所以即便为稀疏输入，输出也可能为稠密向量。

下面的示例展示如果读入一个libsvm形式的数据以及调整其特征值到[0,1]之间。
  *
  */
object MinMaxScalerTest1 {

    def main(args: Array[String]): Unit = {
      val sparkSession = SparkSession.builder()
        .appName("MinMaxScaler")
        .master("local[2]")
        .getOrCreate()
      val data = sparkSession.read.format("libsvm").load("./data/mllib/sample_libsvm_data.txt")

      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")

      // Compute summary statistics and generate MinMaxScalerModel
      val scalerModel = scaler.fit(data)

      // rescale each feature to range [min, max].
      val scaledData = scalerModel.transform(data)
      scaledData.show()

    }
}
