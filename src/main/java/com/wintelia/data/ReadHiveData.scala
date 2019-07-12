package com.wintelia.data

import java.sql.DriverManager

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer


object ReadHiveData {
  def main(args: Array[String]): Unit = {

    Class.forName("org.apache.hive.jdbc.HiveDriver")
    val conn = DriverManager.getConnection("jdbc:hive2://10.108.1.56:10000/", "root", "hadoop")

    try {
      val statement = conn.createStatement
      statement.execute("use test")
      val rs = statement.executeQuery("select * from flight_fips ")

      var arrB = new ArrayBuffer[Vector]()
      while (rs.next()) {
        val id = rs.getString("id")
        val flight_id = rs.getString("flight_id")
        val route = rs.getString("route")
        val reg_no = rs.getString("reg_no")

        arrB += Vectors.dense(id.length,flight_id.length, route.length, reg_no.length)

      }

      val conf = new SparkConf().setAppName("PCAExample").setMaster("local[8]")
      val sc = new SparkContext(conf)

      val arr = arrB.toArray
      val data: RDD[Vector] = sc.parallelize(arr)

      // PCA降维
      val pca: RDD[Vector] = new PCA(2).fit(data).transform(data)
      pca.foreach(println)
    } catch {
      case e: Exception => e.printStackTrace
    }



  }
}
