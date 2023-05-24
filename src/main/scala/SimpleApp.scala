/* SimpleApp.scala */

import org.apache.spark.{SparkConf, SparkContext}
import play.api.libs.json.Json

import java.io.{File, PrintWriter}
import scala.collection.mutable

object SimpleApp {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("sparkbyexamples.com").setMaster("local[1]")
    val sparkContext = new SparkContext(conf)
    val lines = sparkContext.textFile(args(0))
    // here each line is a json doc
    val getAllCat = lines
      .map(line => Json.parse(line))
      .map(line => {
        val cat = (line \ "category").get.toString()
        (cat, 1)
      })
      .reduceByKey(_ + _)

    val mapWithAllCategories = getAllCat.collectAsMap()

    val sumAll = mapWithAllCategories.values.sum
    val numAs = lines
      .map(line => Json.parse(line))
      .flatMap(jsonLine => {
        val category = (jsonLine \ "category").get.toString()
        val delimitersRegex = """[\s\t\d\(\)\[\]\{\}.!?,;:+=\-_"'`~#@&*%€$§\\\/]+"""
        val sets = (jsonLine \ "reviewText").get.toString().split(delimitersRegex).toSet
        sets
          .filter(word => word.length != 1)
          .map(word => (word -> category) -> 1)
      })
      .reduceByKey(_ + _)
      .groupBy(f => f._1._1)
      .map(f => f._1 -> f._2.map(ff => ff._1._2 -> ff._2))

    val res = numAs
      .flatMap(f => {
        val sumOverAllCat = f._2.map(ff => ff._2).sum;
        // calculate the chi-square
        f._2.map(pair => {
          val A: Double = pair._2
          val B: Double = sumOverAllCat - A
          val C: Double = mapWithAllCategories(pair._1) - A
          val D: Double = sumAll - mapWithAllCategories(pair._1) - B
          var r = ((A * D - B * C) * (A * D - B * C)) / ((A + B) * (A + C) * (B + D) * (C + D));
          r = sumAll * r
          pair._1 -> (f._1 -> r)
        })
      })
      .groupByKey()
      .map(f => {
        val pq = mutable.PriorityQueue[(String, Double)]()(Ordering.by[(String, Double), Double](_._2).reverse)
        f._2.
          map(p => {
            pq.enqueue(p)
            if (pq.size > 75)
              pq.dequeue()
          })
        f._1 -> pq.dequeueAll.reverse.toList
      })

    res.foreach(r => {
      println(s"${r._1}")
      r._2.foreach(p => print(s"${p._1}:${p._2} "))
      println()
    })

    val p = res.map(f => f._2.map(ff => ff._1)).reduce(_ ++ _)
    // this is the line containing the whole dictionary
    println(p.sorted.mkString("Array(", ", ", ")"))

    val outputPath = "/Users/casparmayrgundter/Documents/SE/SoSe23/DIC/Exercise2/output.txt"

    val writer = new PrintWriter(outputPath)
    try {
      writer.println(p.sorted.mkString("Array(", ",", ")"))
    } finally {
      writer.close()
    }

    sparkContext.stop()
  }
}