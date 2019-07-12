package com.sparkml.mytest

import java.util

import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{DataTypes, StructField}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
/**
TF（词频Term Frequency）：HashingTF与CountVectorizer都可以用于生成词频TF矢量。

HashingTF是一个转换器（Transformer），它可以将特征词组转换成给定长度的（词频）特征向量组。
在文本处理中，“特征词组”有一系列的特征词构成。HashingTF利用hashing trick将原始的特征（raw feature）通过哈希函数映射到
低维向量的索引（index）中。这里使用的哈希函数是murmurHash 3。词频（TF）是通过映射后的低维向量计算获得。
通过这种方法避免了直接计算（通过特征词建立向量term-to-index产生的）巨大特征数量。
（直接计算term-to-index 向量）对一个比较大的语料库的计算来说开销是非常巨大的。
但这种降维方法也可能存在哈希冲突：不同的原始特征通过哈希函数后得到相同的值（ f(x1) = f(x2) ）。
为了降低出现哈希冲突的概率，我们可以增大哈希值的特征维度，例如：增加哈希表中的bucket的数量。
一个简单的例子：通过哈希函数变换到列的索引，这种方法适用于2的幂（函数）作为特征维度，否则（采用其他的映射方法）就会出现特征不能
均匀地映射到哈希值上。默认的特征维度是 218=262,144218=262,144 。一个可选的二进制切换参数控制词频计数。
当设置为true时，所有非零词频设置为1。这对离散的二进制概率模型计算非常有用。

CountVectorizer可以将文本文档转换成关键词的向量集。请阅读英文原文CountVectorizer 了解更多详情。

IDF（逆文档频率）：IDF是的权重评估器（Estimator），用于对数据集产生相应的IDFModel（不同的词频对应不同的权重）。
IDFModel对特征向量集（一般由HashingTF或CountVectorizer产生）做取对数（log）处理。
直观地看，特征词出现的文档越多，权重越低（down-weights colume）。
  */
object TFIDF {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val sc = new SparkContext(conf)
    val sql = new SQLContext(sc);

    val fields = new util.ArrayList[StructField];
    fields.add(DataTypes.createStructField("label", DataTypes.DoubleType, true));
    fields.add(DataTypes.createStructField("sentence", DataTypes.StringType, true));
    val data = sc.parallelize(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    ))
    val structType = DataTypes.createStructType(fields);

    val row = data.map {
      row => Row(row._1, row._2)
    }
    val df: DataFrame = sql.createDataFrame(row, structType)
    df.printSchema()
    df.show()
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(df)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show(false)
  }

}
