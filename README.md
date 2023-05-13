# spark_exercise_2
solution for exec2
## how to run
`sbt assemble`

after this submit the job to apache spark

`{path_to_spark}/bin/spark-submit --class "SimpleApp" --master local[1] target/scala-2.13/spark-assembly-0.1.0-SNAPSHOT.jar {path_to_test_dataset}/reviews_devset.json`

you need **scala 2.13.10** (+ sbt just get them using sdk man) and **apache spark 3.4.0**

## note 
the solution is not yet written to a file, so this should also be done

it is also not yet tested on the cluster
