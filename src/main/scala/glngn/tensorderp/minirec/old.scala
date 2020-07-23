package glngn.tensorderp.minirec

import org.platanios.tensorflow.api._
import zio._

object old {
/*
  pick one [cheese]:

  pick one [bacon]: cheese
  History: "cheese"
  pick one [bacon]: cheese
  History: "cheese", "cheese"
  pick one [cheese]: cheese
  */
  object SampleData {
    val choices0 = List("bacon", "cheese", "tomato", "more cheese", "veggies", "nothing")
  }

  class Recommender(choices: Seq[String]) {
    object model {

      object inputs

      object variables {
        val estimatedDist = tf.variable[Float]("estimateDist",
                                              Shape(choices.size),
                                              tf.RandomUniformInitializer())
      }

      object outputs {
        val choice = tf.argmax(variables.estimatedDist.toOutput, 0, INT32)
      }

      def loss(expected: Int): Output[Float] = {
        val expectedDist = tf.oneHot(expected, choices.size, 1f, 0f)
        tf.mean(tf.softmaxCrossEntropy(variables.estimatedDist.toOutput,expectedDist))
      }
    }

    val session = core.client.Session()
    session.run(targets = model.variables.estimatedDist.initializer)

    // give a recommendation - predict the most likley
    def predict: String = {
      val predictedChoice: Int = session.run(fetches = model.outputs.choice).scalar
      choices(predictedChoice)
    }

    val trainer = ops.training.optimizers.GradientDescent(learningRate = 0.1f)

    // update the estimated distribution - train
    def train(expected: String): Unit = {
      val expectedIndex = choices.indexOf(expected)
      val loss = model.loss(expectedIndex)
      val updates = trainer.minimize(loss)
      session.run(targets = updates)
    }
  }
}
