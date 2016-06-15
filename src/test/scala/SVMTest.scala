import breeze.linalg.{DenseVector => BreezeDense}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.feature.{PCA, StandardScalerModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Before, Test}

/**
  * Created by Christian on 05.06.2016.
  */


@Test
object SVMTest {
  @Before
  def prepare(): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Users\\Christian\\Dev\\hadoop-2.6.0")
  }

  def flattenPclass(value: Double): (Double, Double, Double) = {
    if (value == 1)
      (1, 0, 0)
    else if (value == 2)
      (0, 1, 0)
    else (0, 0, 1)
  }


  def flattenEmbarked(value: Double): (Double, Double, Double) = {
    if (value == 0)
      (1, 0, 0)
    else if (value == 1)
      (0, 1, 0)
    else (0, 0, 1)
  }

  def flattenSex(value: Double): (Double, Double) = {
    if (value == 0)
      (1, 0)
    else (0, 1)
  }

  def scaleValue(min: Double, max: Double, value: Double): Double = {
    (value - min) / max - min
  }

  def getScaledVector(fare: Double, age: Double, pclass: Double, sex: Double, embarked: Double, scaler: StandardScalerModel): org.apache.spark.mllib.linalg.Vector = {
    val scaledContinous = scaler.transform(Vectors.dense(fare, age))
    val pclassFlat: (Double, Double, Double) = flattenPclass(pclass)
    val embarkedFlat: (Double, Double, Double) = flattenEmbarked(embarked)
    val sexFlat: (Double, Double) = flattenSex(sex)
    Vectors.dense(
      scaledContinous(0),
      scaledContinous(1),
      sexFlat._1,
      sexFlat._2,
      pclassFlat._1,
      pclassFlat._2,
      pclassFlat._3,
      embarkedFlat._1,
      embarkedFlat._2,
      embarkedFlat._3)
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
    val stddev = Vectors.dense(math.sqrt(summary.variance(0)), math.sqrt(summary.variance(1)))
    val mean = Vectors.dense(summary.mean(0), summary.mean(1))
    val scaler = new StandardScalerModel(stddev, mean)

    val scaledData = trainingDf.map { row =>
      LabeledPoint(row.getAs[Int]("Survived"),
        getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler))
    }

    val pca = new PCA(scaledData.first().features.size).fit(scaledData.map(_.features))
    val pcaData =scaledData.map{
      lpoint => LabeledPoint(lpoint.label, pca.transform(lpoint.features))
    }

    //scaledData.saveAsTextFile("results/vectors")
    val model: SVMModel = SVMWithSGD.train(pcaData, 100)
    //val naiveBayesModel = NaiveBayes.train(scaledData, lambda = 1.0, modelType = "multinomial")
//    val lrmodel = new LogisticRegressionWithLBFGS()
//      .setNumClasses(2)
//      .run(scaledData)

    val validationDf: DataFrame = InOutUtil.getValidationDf(sqlContext)

    val resultRDD = validationDf.map { row =>
      val pclassFlat: (Double, Double, Double) = flattenPclass(row.getAs[Int]("Pclass"))
      val embarkedFlat: (Double, Double, Double) = flattenEmbarked(row.getAs[Int]("Embarked"))
      val denseVecor = getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler)
      val pcaVector = pca.transform(denseVecor)
      val result = model.predict(pcaVector)
      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))
    }

//    val resultRDDLR = validationDf.map { row =>
//      val pclassFlat: (Double, Double, Double) = flattenPclass(row.getAs[Int]("Pclass"))
//      val embarkedFlat: (Double, Double, Double) = flattenEmbarked(row.getAs[Int]("Embarked"))
//      val denseVecor = getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler)
//      val result = lrmodel.predict(denseVecor)
//      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))
//    }

    /*val resultRDDBayes = validationDf.map { row =>
      val pclassFlat: (Double, Double, Double) = flattenPclass(row.getAs[Int]("Pclass"))
      val embarkedFlat: (Double, Double, Double) = flattenEmbarked(row.getAs[Int]("Embarked"))
      val denseVecor = getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), summary)
      val result = naiveBayesModel.predict(denseVecor)
      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))    }*/

    InOutUtil.saveResult("SVM_pca", sqlContext, resultRDD)
    //InOutUtil.saveResult("LogisticRegression", sqlContext, resultRDDLR)
    //InOutUtil.saveResult("SVM_bayes", sqlContext, resultRDDBayes)


  }

}
