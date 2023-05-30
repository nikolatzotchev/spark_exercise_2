# spark_exercise_2
solution for exec2
## how to run
`sbt assembly`

after this submit the job to apache spark

* Part 1
`{path_to_spark}/bin/spark-submit --class "SimpleApp" --master local[1] target/scala-2.13/spark-assembly-0.1.0-SNAPSHOT.jar {path_to_test_dataset}/reviews_devset.json`

* Part 2
`{path_to_spark}/bin/spark-submit --class "Pipelines" --master local[1] target/scala-2.13/spark-assembly-0.1.0-SNAPSHOT.jar {path_to_test_dataset}/reviews_devset.json {path_to_stopwords}/stopwords.txt {path_to_output}/stopwords.txt`

* Part 3
`{path_to_spark}/bin/spark-submit --class "Classification" --master local[1] target/scala-2.13/spark-assembly-0.1.0-SNAPSHOT.jar {path_to_test_dataset}/reviews_devset.json {path_to_stopwords}/stopwords.txt`


you need **scala 2.13.10** (+ sbt just get them both using sdk man [sdkman](https://sdkman.io/)) and **apache spark 3.4.0** [download link](https://spark.apache.org/downloads.html) get the one with scala 2.13.10

## note 
the solution is not yet written to a file, so this should also be done

it is also not yet tested on the cluster
