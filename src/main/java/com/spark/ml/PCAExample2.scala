package com.spark.ml

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD



object PCAExample2 {

  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("PCAExample").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //sc.setLogLevel("WARN")

    val data = Array(
      Vectors.dense(4.0,1.0,4.0,5.0),
      Vectors.dense(2.0,3.0, 4.0, 5.0),
      Vectors.dense(4.0,0.0, 6.0, 7.0))

    /*mllib包方式*/
    var arr = Array(
      Vectors.dense(4.0,1.0,4.0,5.0),
      Vectors.dense(2.0,3.0, 4.0, 5.0),
      Vectors.dense(4.0,0.0, 6.0, 7.0))
    val data1: RDD[Vector] = sc.parallelize(arr)
    val pca: RDD[Vector] = new PCA(2).fit(data1).transform(data1)
    pca.foreach(println)

    /*协方差计算方式*/
    val mat : RowMatrix = new RowMatrix(sc.parallelize(data))
    val pc = mat.computePrincipalComponents(2);
    val projected: RowMatrix = mat.multiply(pc)
    projected.rows.foreach(println)

  }

}
