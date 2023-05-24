/* SimpleApp.scala */

import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel, CountVectorizer, HashingTF, IDF, IndexToString, RegexTokenizer, StopWordsRemover, StringIndexer, UnivariateFeatureSelector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
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

    val sqlContext = new SQLContext(sparkContext)
    val df: DataFrame = sqlContext.read.json(args(0))

    val stopWordsRDD = sparkContext.textFile("/Users/casparmayrgundter/Documents/SE/SoSe23/DIC/Exercise2/stopwords.txt")
    val stopWords: Array[String] = stopWordsRDD.collect()


    val labelIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W")
      .setInputCol("reviewText")
      .setOutputCol("tokens")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filteredTokens")
      .setCaseSensitive(false)
      .setStopWords(stopWords)

    val countVectorizer = new CountVectorizer()
      .setInputCol("filteredTokens")
      .setOutputCol("countedTokens")

    val chiSqSelector = new UnivariateFeatureSelector()
      .setFeatureType("categorical")
      .setLabelType("categorical")
      .setFeaturesCol("countedTokens")
      .setLabelCol("categoryIndex")
      .setOutputCol("selectedFeatures")
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(2000)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, tokenizer, stopWordsRemover, countVectorizer, chiSqSelector))

    val model = pipeline.fit(df)

    val data = model.transform(df)

    val selectedFeatures = data.select("selectedFeatures")

    val x = data.head()
    println(x)

    data.show()

    /*
    val selectedTermsFile = "/Users/casparmayrgundter/Documents/SE/SoSe23/DIC/Exercise2/output_ds.txt"
    sparkContext.parallelize(selectedTerms).saveAsTextFile(selectedTermsFile)*/

    sparkContext.stop()
  }
}