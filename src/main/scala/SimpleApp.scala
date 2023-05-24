/* SimpleApp.scala */

import org.apache.spark.{SparkConf, SparkContext}
import play.api.libs.json.Json

import java.io.{File, PrintWriter}
import java.util.stream.Collectors
import scala.collection.mutable

object SimpleApp {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("sparkbyexamples.com").setMaster("local[1]")
    val sparkContext = new SparkContext(conf)
    val lines = sparkContext.textFile(args(0))

    // this contains for each category the number of reviews
    val getAllCat = lines
      .map(line => Json.parse(line))
      .map(line => {
        val category = (line \ "category").get.toString()
        (category, 1)
      })
      .reduceByKey(_ + _)

    val mapWithAllCategories = getAllCat.collectAsMap()

    val sumAll = mapWithAllCategories.values.sum

    val numAs = lines
      .map(line => Json.parse(line))
      .flatMap(jsonLine => {
        val category = (jsonLine \ "category").get.toString()
        val delimitersRegex = "[^a-zA-Z]+"
        val sets = (jsonLine \ "reviewText").get.toString().split(delimitersRegex).toSet
        sets
          // remove one character words
          .filter(word => word.length != 1)
          // each entry will contain the (word, category), 1 so that we can count the words in the different categories
          .map(word => (word -> category) -> 1)
      })
      // sum and group by key
      .reduceByKey(_ + _)
      // here the group is done by word, and not by (word, category) so that we can have for each word all of the sums categories
      .groupBy(f => f._1._1)
      // here we map the categories
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
      .sortByKey()

    val stringBuilder = new StringBuilder()

    val p = res.map(f => f._2.map(ff => ff._1)).reduce(_ ++ _)
    val result = res.map { case (key, array) =>
      key -> array.map { case (subKey, value) =>
        s"$subKey:$value"
      }.mkString(" ")
    }
    val finalString = result.map(r => s"${r._1}\t ${r._2}").collect().mkString("\n")

    stringBuilder.append(finalString).append("\n")

    stringBuilder.append(p.sorted.mkString(" "))

    val outputPath = "output_rdd.txt"

    val writer = new PrintWriter(outputPath)
    try {
      writer.println(stringBuilder.toString())
    } finally {
      writer.close()
    }
    sparkContext.stop()
  }
}