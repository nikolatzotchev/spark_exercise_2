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


    // our categories are strings. ChiSqSelector can't handle these - therefore thtey have to be mapped to indexes
    val labelIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)

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

    // builds the pipeline with all the stages that were previously created
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, tokenizer, stopWordsRemover, countVectorizer, chiSqSelector))

    val model = pipeline.fit(df)

    model.transform(df).show()

    // get the vocabulary
    val vocab = model.stages(3).asInstanceOf[CountVectorizerModel].vocabulary

    // get the 200 top features
    val selectedFeatures = model.stages.last.asInstanceOf[UnivariateFeatureSelectorModel].selectedFeatures

    // map the indexes of the top features to the actual words
    val words = selectedFeatures.map(index => vocab(index))

    //export the words
    val selectedTermsFile = "/Users/casparmayrgundter/Documents/SE/SoSe23/DIC/Exercise2/output_ds.txt"
    sparkContext.parallelize(words.sorted).saveAsTextFile(selectedTermsFile)

    sparkContext.stop()
  }
}