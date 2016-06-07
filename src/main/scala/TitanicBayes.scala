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
      (row.getAs[Int]("Survived"), row.getAs[Double]("Fare"), row.getAs[Int]("Pclass"), row.getAs[Double]("Age")
        ,row.getAs[Int]("Sex"), row.getAs[Int]("Parch"), row.getAs[Int]("SibSp"),row.getAs[Int]("Embarked")))

    val labledData = mappedDf.map { case (survived, fare, pclass, age, sex, parch, sibsp, embarked) =>
      LabeledPoint(survived, Vectors.dense(fare, pclass, age, sex, parch, sibsp, embarked))
    }
    naiveBayesModel = NaiveBayes.train(labledData, lambda = 1.0, modelType = "multinomial")

  }

  def predict(df: DataFrame): RDD[Row] = {

    val resultDf = df.map { row =>
      val denseVecor = Vectors.dense(row.getAs[Double]("Fare"), row.getAs[Int]("Pclass"), row.getAs[Double]("Age"),row.getAs[Int]("Sex"),
        row.getAs[Int]("Parch"), row.getAs[Int]("SibSp"), row.getAs[Int]("Embarked") )
      val result = naiveBayesModel.predict(denseVecor)
      Row.fromTuple((row.getAs[Int]("PassengerId"), result.toInt))
    }
    resultDf
  }


}
