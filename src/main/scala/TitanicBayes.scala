import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Created by christian.freischlag on 17.02.2016.
  */
object TitanicBayes {

  var naiveBayesModel: NaiveBayesModel = null

  def train(df: DataFrame): Unit = {
    val mappedDf = df.map(row =>
      (row.getAs[Int]("Survived"), row.getAs[Double]("Fare"), row.getAs[Int]("Pclass"), row.getAs[Double]("Age")))

    val labledData = mappedDf.map { case (survived, fare, pclass, age) =>
      LabeledPoint(survived, Vectors.dense(fare, pclass, age))
    }
    naiveBayesModel = NaiveBayes.train(labledData, lambda = 5.0, modelType = "multinomial")

  }

  def predict(df: DataFrame): RDD[Row] = {

    val resultDf = df.map { row =>
      val denseVecor = Vectors.dense(row.getAs[Double]("Fare"), row.getAs[Int]("Pclass"), row.getAs[Double]("Age"))
      val result = naiveBayesModel.predict(denseVecor)
      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))
    }
    resultDf
  }


}
