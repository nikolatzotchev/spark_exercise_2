
lazy val root = (project in file("."))
  .settings(
    name := "spark",
    scalaVersion := "2.13.10",
    version := "0.1.0-SNAPSHOT",
    libraryDependencies += "org.apache.spark" %% "spark-core" % "3.4.0",
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.3.2",
    libraryDependencies += "com.typesafe.play" %% "play-json" % "2.9.4"

  )
ThisBuild / assemblyMergeStrategy := {
    case PathList("META-INF", _*) => MergeStrategy.discard
    case _ => MergeStrategy.first
}
