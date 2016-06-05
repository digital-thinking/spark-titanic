import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Before, Test}
import org.apache.spark.sql.functions._

/**
  * Created by Christian on 05.06.2016.
  */


@Test
object TitanicBayesTest {
  @Before
  def prepare(): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Users\\Christian\\Dev\\hadoop-2.6.0")
  }


  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("TrendyWords_" + System.currentTimeMillis())
    conf.set("spark.executor.memory", "4G")
    conf.set("spark.executor.cores", "4")
    conf.set("spark.master", "local[4]")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryoserializer.buffer.max", "512m")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)


    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("src/main/resources/titanic_data/train.csv")


    // Get statistics about Fare and Age
    val statsDf = df.map { row =>
      Vectors.dense(row.getAs("Fare"), row.getAs("Age"))
    }
    val summary: MultivariateStatisticalSummary = Statistics.colStats(statsDf)

    val meanFare =  summary.mean(0)
    val meanAge =  summary.mean(1)

    // Define udfs, which fill in mean values, if the data has no entry
    val normFare = udf((d: String) => d match {
      case null => Some(meanFare)
      case s => Some(s.toDouble)
    })

    val normAge = udf((d: String) => d match {
      case null => Some(meanAge)
      case s => Some(s.toDouble)
    })

    // select the columns we need and apply the udfs
    val preprocessed = df.select(df("Survived"), normFare(df("Fare")).alias("Fare"), normAge(df("Age")).alias("Age"), df("Pclass"))

    // Create a Model
    TitanicBayes.train(preprocessed)

    // load the Test data
    val testDf = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("src/main/resources/titanic_data/test.csv")

    // Get predictions from the model
    val result = TitanicBayes.predict(testDf)

    // convert the RDD to Dataframe and save the Data to Disk
    val customSchema = StructType(Array(
      StructField("PassengerId", IntegerType, false),
      StructField("Survived", IntegerType, false))
    )
    val resultDf = sqlContext.createDataFrame(result, customSchema)

    resultDf
      //  only one file (one partition)
      .coalesce(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save("results/out_" + System.currentTimeMillis())


  }

}
