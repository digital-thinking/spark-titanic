import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Before, Test}

/**
  * Created by Christian on 05.06.2016.
  */


@Test
object DecisionTrees {
  @Before
  def prepare(): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Users\\Christian\\Dev\\hadoop-2.6.0")
  }

  def scaleValue(min: Double, max: Double, value: Double): Double = {
    (value - min) / max - min
  }


  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Naive_bayes_titanic")
    conf.set("spark.master", "local[4]")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryoserializer.buffer.max", "512m")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val trainingDf: DataFrame = Util.getTrainingDf(sqlContext, true)
    val summary = Util.summary
    val stddev = Vectors.dense(math.sqrt(summary.variance(0)), math.sqrt(summary.variance(1)))
    val mean = Vectors.dense(summary.mean(0), summary.mean(1))
    val scaler = new StandardScalerModel(stddev, mean)

    val scaledData = trainingDf.map { row =>
      LabeledPoint(row.getAs[Int]("Survived"),
        Util.getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler))
    }

    val numClasses = 2
    //val categoricalFeaturesInfo = Map[Int, Int]((2, 3), (3, 2), (4, 3))
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 96
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 100

    val model = RandomForest.trainClassifier(scaledData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    //scaledData.saveAsTextFile("results/vectors")


    val validationDf: DataFrame = Util.getValidationDf(sqlContext)

    val resultRDD = validationDf.map { row =>
      val denseVecor = Util.getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler)
      val result = model.predict(denseVecor)
      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))
    }



    Util.saveResult("RandomForest", sqlContext, resultRDD)


  }

}
