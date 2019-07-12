package com.wintelia.data

import org.apache.spark.sql.SparkSession


object FLightDelayData {

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder()
      .appName("FlightDelay")
      .master("local[2]")
      .getOrCreate()
    val flightData = sparkSession.read.format("com.databricks.spark.csv").option("header","true").load("./data/flight/1998.csv")
    flightData.registerTempTable("flights")

    val airportData = sparkSession.read.format("com.databricks.spark.csv").option("header","true").load("./data/flight/airports.csv")
    airportData.registerTempTable("airports")

    val queryFlightNumResult = sparkSession.sql("SELECT COUNT(FlightNum) FROM flights WHERE DepTime BETWEEN 0 AND 600")

    println(queryFlightNumResult.take(1))

    // COUNT(DISTINCT DayofMonth) 的作用是计算每个月的天数
    val queryFlightNumResult1 = sparkSession.sql("SELECT COUNT(FlightNum)/COUNT(DISTINCT DayofMonth) FROM flights WHERE Month = 1 AND DepTime BETWEEN 1001 AND 1400")

    println(queryFlightNumResult1.take(1))

    val queryDestResult = sparkSession.sql("SELECT DISTINCT Dest, ArrDelay FROM flights WHERE ArrDelay = 0")

    println(queryDestResult.take(1))

  }
}
