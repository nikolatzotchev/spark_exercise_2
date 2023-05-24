/* SimpleApp.scala */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import play.api.libs.json.Json

import java.io.PrintWriter
import scala.collection.mutable

object Classification {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("sparkbyexamples.com").setMaster("local[1]")
    val sparkContext = new SparkContext(conf)

    val sqlContext = new SQLContext(sparkContext)
    val df: DataFrame = sqlContext.read.json(args(0))

    val stopWordsRDD = sparkContext.textFile("/Users/casparmayrgundter/Documents/SE/SoSe23/DIC/Exercise2/stopwords.txt")
    val stopWords: Array[String] = stopWordsRDD.collect()

    val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2), seed = 42)

    val labelIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      //.fit(splits(0))

    val tokenizer = new RegexTokenizer()
      .setPattern("[^[a-zA-Z]]")
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

    val normalizer = new Normalizer()
      .setInputCol("selectedFeatures")
      .setOutputCol("normalizedSelectedFeatures")

    val lsvm = new LinearSVC()
      .setFeaturesCol("normalizedSelectedFeatures")
      .setLabelCol("categoryIndex")
      .setMaxIter(100)
      .setRegParam(0.001)

    val ovr = new OneVsRest()
      .setClassifier(lsvm)
      .setFeaturesCol("normalizedSelectedFeatures")
      .setLabelCol("categoryIndex")
      .setPredictionCol("predictions")

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, tokenizer, stopWordsRemover, countVectorizer, chiSqSelector, normalizer, ovr))


    val gridSearch = new ParamGridBuilder()
      .addGrid(lsvm.regParam, Array(1.0, 0.1, 0.001))
      .addGrid(lsvm.standardization, Array(true, false))
      .addGrid(lsvm.maxIter, Array(10, 100))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setPredictionCol("predictions")
      .setLabelCol("categoryIndex")

    val tv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(gridSearch)
      .setTrainRatio(0.8)

    //val model = pipeline.fit(trainData)

    val validationModel = tv.fit(trainData)

    val predictions = validationModel.transform(testData)

    //val normalizedDf = normalizer.transform(tmp)

    //val predictions = ovr.fit(normalizedDf).transform(normalizedDf)

    predictions.show()

    // compute the classification error on test data.
    val accuracy = evaluator.evaluate(predictions)
    println(s"F1 score = ${accuracy}")

    sparkContext.stop()
  }
}