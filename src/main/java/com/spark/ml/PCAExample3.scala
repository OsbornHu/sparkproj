package com.spark.ml

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.PCA
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.ml.linalg.Vectors


object PCAExample3 {
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("PCAExample").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //sc.setLogLevel("WARN")

    val data = Array(
      Vectors.dense(4.0,1.0,4.0,5.0),
      Vectors.dense(2.0,3.0, 4.0, 5.0),
      Vectors.dense(4.0,0.0, 6.0, 7.0))

    /*  //mllib包
    val data = sc.parallelize(arr)
    val pca = new PCA(2).fit(data).transform(data)
    pca.foreach(println)

    val mat : RowMatrix = new RowMatrix(sc.parallelize(data))
    val pc = mat.computePrincipalComponents(2);
    val projected: RowMatrix = mat.multiply(pc)
    projected.rows.foreach(println)*/

    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    //标准化
    val data2 = new StandardScaler().setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(false)
      .setWithMean(true)
      .fit(df)
      .transform(df)

    val pca = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")
      .setK(2)
      .fit(data2)

    val pcaDF = pca.transform(data2)
    val result = pcaDF.select("pcaFeatures")
    result.show(100,false);

    //选择k值
    var i=1;
    for( x <- pca.explainedVariance.toArray ){
      println(i+"\t"+x)
      i+=1
    }

  }
}
