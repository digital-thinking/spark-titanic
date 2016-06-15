import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
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

  def getScaledVector(fare: Double, age: Double, pclass: Double, sex: Double, embarked: Double, summary: MultivariateStatisticalSummary): org.apache.spark.mllib.linalg.Vector = {
    Vectors.dense(
      scaleValue(summary.min(0), summary.max(0), fare),
      scaleValue(summary.min(1), summary.max(1), age),
      pclass - 1,
      sex,
      embarked
    )
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Naive_bayes_titanic")
    conf.set("spark.master", "local[4]")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryoserializer.buffer.max", "512m")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val trainingDf: DataFrame = InOutUtil.getTrainingDf(sqlContext, true)
    val summary = InOutUtil.summary

    val scaledData = trainingDf.map { row =>
      LabeledPoint(row.getAs[Int]("Survived"),
        getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), summary))
    }

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]((2, 3), (3, 2), (4, 3))
    val numTrees = 100
    val featureSubsetStrategy = "all"
    val impurity = "entropy"
    val maxDepth = 5
    val maxBins = 200

    val model = RandomForest.trainClassifier(scaledData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    //scaledData.saveAsTextFile("results/vectors")


    // GradientBoosted
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 10 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxBins = 200
    // Empty categoricalFeaturesInfo indicates s features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = categoricalFeaturesInfo

    val model2 = GradientBoostedTrees.train(scaledData, boostingStrategy)


    val validationDf: DataFrame = InOutUtil.getValidationDf(sqlContext)

    val resultRDD = validationDf.map { row =>
      val pclassFlat: (Double, Double, Double) = SVMTest.flattenPclass(row.getAs[Int]("Pclass"))
      val embarkedFlat: (Double, Double, Double) = SVMTest.flattenEmbarked(row.getAs[Int]("Embarked"))
      val denseVecor = getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), summary)
      val result = model.predict(denseVecor)
      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))
    }

    val resultRDD2 = validationDf.map { row =>
      val pclassFlat: (Double, Double, Double) = SVMTest.flattenPclass(row.getAs[Int]("Pclass"))
      val embarkedFlat: (Double, Double, Double) = SVMTest.flattenEmbarked(row.getAs[Int]("Embarked"))
      val denseVecor = getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), summary)
      val result = model2.predict(denseVecor)
      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))
    }

    InOutUtil.saveResult("RandomForest", sqlContext, resultRDD)
    InOutUtil.saveResult("GradientBoostedTrees", sqlContext, resultRDD2)


  }

}
