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





  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Naive_bayes_titanic")
    conf.set("spark.master", "local[4]")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryoserializer.buffer.max", "512m")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val trainingDf: DataFrame = Util.getTrainingDf(sqlContext, true)

    val r = scala.util.Random
    trainingDf.rdd.keyBy(_.getAs("IP").toString).map{case (key, value) => (key+r.nextInt(10), value)}

    val summary = Util.summary
    val stddev = Vectors.dense(math.sqrt(summary.variance(0)), math.sqrt(summary.variance(1)))
    val mean = Vectors.dense(summary.mean(0), summary.mean(1))
    val scaler = new StandardScalerModel(stddev, mean)

    val scaledData = trainingDf.map { row =>
      LabeledPoint(row.getAs[Int]("Survived"),
        Util.getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler))
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

    val validationDf: DataFrame = Util.getValidationDf(sqlContext)

    val resultRDD = validationDf.map { row =>
      val denseVecor = Util.getScaledVector(row.getAs[Double]("Fare"), row.getAs[Double]("Age"), row.getAs[Int]("Pclass"), row.getAs[Int]("Sex"), row.getAs[Int]("Embarked"), scaler)
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

    Util.saveResult("SVM_pca", sqlContext, resultRDD)
    //InOutUtil.saveResult("LogisticRegression", sqlContext, resultRDDLR)
    //InOutUtil.saveResult("SVM_bayes", sqlContext, resultRDDBayes)


  }

}
