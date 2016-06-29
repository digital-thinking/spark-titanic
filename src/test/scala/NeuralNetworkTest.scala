import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Before, Test}


/**
  * Created by Christian on 05.06.2016.
  */


@Test
object NeuralNetworkTest {
  @Before
  def prepare(): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Users\\Christian\\Dev\\hadoop-2.6.0")
  }

  def scaleValue(min: Double, max: Double, value: Double): Double = {
    (value - min) / max - min
  }

  case class Feature(v: Vector)

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Neural_network_titanic")
    conf.set("spark.master", "local[4]")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryoserializer.buffer.max", "512m")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val trainingDf: DataFrame = Util.getTrainingDf(sqlContext, true)
    val summary = Util.summary
    val stddev = Vectors.dense(math.sqrt(summary.variance(0)), math.sqrt(summary.variance(1)))
    val mean = Vectors.dense(summary.mean(0), summary.mean(1))
    val scaler = new StandardScalerModel(stddev, mean)

    val scaledData = trainingDf.map { row =>
      (row.getAs[Int]("Survived").toDouble,
        Util.getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler))
    }
    val data: DataFrame = scaledData.toDF("label", "features")

    val layers = Array[Int](10, 8, 7, 4, 2)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setTol(1E-6)
      .setMaxIter(500)

    val model = trainer.fit(data)
    //scaledData.saveAsTextFile("results/vectors")
    val validationDf: DataFrame = Util.getValidationDf(sqlContext)
    val vectors = validationDf.map { row =>
      (row.getAs[Int]("PassengerId"), Util.getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler))
    }.toDF("PassengerId", "features")

    val predicted: DataFrame = model.transform(vectors)



    Util.saveResult("NeuralNetwork", sqlContext, predicted.select(predicted("PassengerId"), predicted("prediction").alias("Survived").cast(IntegerType)).rdd)
    //    predicted.write.format("com.databricks.spark.csv")
    //      .option("header", "true") // Use first line of all files as header
    //      .option("inferSchema", "true") // Automatically infer data types
    //      .save("results/NeuralNetwork_" + System.currentTimeMillis())

  }

}