/* SimpleApp.scala */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object Classification {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("sparkbyexamples.com").setMaster("local[1]")
    val sparkContext = new SparkContext(conf)

    val sqlContext = new SQLContext(sparkContext)
    val df: DataFrame = sqlContext.read.json(args(0))

    val stopWordsRDD = sparkContext.textFile(args(1))
    val stopWords: Array[String] = stopWordsRDD.collect()

    // split the data into train and test data
    // validation will be later split from train data
    val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2), seed = 42)

    // our categories are strings. ChiSqSelector can't handle these - therefore thtey have to be mapped to indexes
    val labelIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    // tokenizer that splits the text into an array of words, only a-z and A-Z are allowed
    val tokenizer = new RegexTokenizer()
      .setPattern("[^[a-zA-Z]]")
      .setInputCol("reviewText")
      .setOutputCol("tokens")

    // removes all stopwords from out txt file
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filteredTokens")
      .setCaseSensitive(false)
      .setStopWords(stopWords)

    // counts the filtered tokens
    val countVectorizer = new CountVectorizer()
      .setInputCol("filteredTokens")
      .setOutputCol("countedTokens")

    // calculates the ChiSq value
    val chiSqSelector = new UnivariateFeatureSelector()
      .setFeatureType("categorical")
      .setLabelType("categorical")
      .setFeaturesCol("countedTokens")
      .setLabelCol("categoryIndex")
      .setOutputCol("selectedFeatures")
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(2000)

    // Normalizes the data
    val normalizer = new Normalizer()
      .setInputCol("selectedFeatures")
      .setOutputCol("normalizedSelectedFeatures")

    // the linear single vector support machine that is used for classification
    // can only handle binary classification
    val lsvm = new LinearSVC()
      .setFeaturesCol("normalizedSelectedFeatures")
      .setLabelCol("categoryIndex")
      .setMaxIter(10)
      .setRegParam(0.1)

    // as we have multi-class classification we need this classifier that performs the multi-class classification
    // with the binary classifier
    val ovr = new OneVsRest()
      .setClassifier(lsvm)
      .setFeaturesCol("normalizedSelectedFeatures")
      .setLabelCol("categoryIndex")
      .setPredictionCol("predictions")

    // this maps the predicted indexes back to our categories
    val indexToString = new IndexToString()
      .setInputCol("predictions")
      .setOutputCol("categoryPredictions")

    // builds the pipeline with all the stages that were previously created
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, tokenizer, stopWordsRemover, countVectorizer, chiSqSelector, normalizer, ovr, indexToString))

    // the assignation of the params with which we want to perform our classification
    val gridSearch = new ParamGridBuilder()
      .addGrid(lsvm.regParam, Array(1.0, 0.1, 0.001))
      .addGrid(lsvm.standardization, Array(true, false))
      .addGrid(lsvm.maxIter, Array(10, 100))
      .build()

    // the evaluator that calculates the f1 scores
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setPredictionCol("predictions")
      .setLabelCol("categoryIndex")

    // splits the train data into train and validation data and performs the gridSearch
    val tv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(gridSearch)
      .setTrainRatio(0.8)

    val model = tv.fit(trainData)

    val predictions = model.bestModel.transform(testData)

    val accuracy = evaluator.evaluate(predictions)
    println(s"F1 score = ${accuracy}")

    sparkContext.stop()
  }
}