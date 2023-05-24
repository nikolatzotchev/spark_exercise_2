/* SimpleApp.scala */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object Pipelines {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("sparkbyexamples.com").setMaster("local[1]")
    val sparkContext = new SparkContext(conf)

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

    val vocab = model.stages(3).asInstanceOf[CountVectorizerModel].vocabulary

    val selectedFeatures = model.stages.last.asInstanceOf[UnivariateFeatureSelectorModel].selectedFeatures

    val words = selectedFeatures.map(index => vocab(index))

    val selectedTermsFile = "/Users/casparmayrgundter/Documents/SE/SoSe23/DIC/Exercise2/output_ds.txt"
    sparkContext.parallelize(words.sorted).saveAsTextFile(selectedTermsFile)

    println("aszgfalsbrfgaks dlgas dgnabsl dkgas")

    sparkContext.stop()
  }
}