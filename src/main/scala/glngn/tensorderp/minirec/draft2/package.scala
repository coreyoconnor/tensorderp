package glngn.tensorderp.minirec.draft2

import org.platanios.tensorflow.api._
import zio._

/**
  * Pick one:
  * 1) choco
  * 2) vanilla
  * 3) caramel
  * 4) more caramel
  *
  * choice: ?
  */
object IceCreamMenu {
  type Flavor = String
  val choices: List[Flavor] = List("chocolate", "vanilla", "caramel", "more caramel")
  val recommender = new Recommender(choices)
}

object ConsoleIceCreamMenu {
  def main(args: Array[String]): Unit = {
    val menu = for {
      _ <- ZIO.foreach(IceCreamMenu.choices) { choice => console.putStrLn(s"- ${choice}") }
      recommendedFlavor <- Recommender.recommend
      _ <- console.putStrLn(s"choice [$recommendedFlavor]: ")
      _ <- Recommender.update("caramel")
    } yield ()

    val io = for {
      _ <- menu.provideCustomLayer(IceCreamMenu.recommender.fresh.toLayer)
      dist <- ZIO.foldLeft(0 to 1000)(Map.empty[String, Int]) { case (dist, _) =>
        val withRecommender =  for {
          _ <- Recommender.update("more caramel")
          flavor <- Recommender.recommend
        } yield {
          dist.get(flavor) match {
            case None => dist + (flavor -> 0)
            case Some(c) => dist + (flavor -> (c + 1))
          }
        }

        withRecommender.provideCustomLayer(IceCreamMenu.recommender.fresh.toLayer)
      }
      _ <- console.putStrLn(s"distribution = ${dist.toList}")
    } yield ()

    Runtime.default.unsafeRun(io)
  }
}

class Recommender(val choices: Seq[String]) {
  def fresh = providers.tensorflow.fresh(this)
}

object Recommender {
  def recommend: ZIO[Has[Service], Throwable, String] =
    ZIO.service[Recommender.Service].flatMap(_.recommend)

  def update(actual: String): ZIO[Has[Service], Throwable, Unit] =
    ZIO.service[Recommender.Service].flatMap(_.update(actual))

  trait Service {
    def recommend: IO[Throwable, String]
    def update(actual: String): IO[Throwable, Unit]
  }
}

object providers {
  object tensorflow {
    trait Model {
      val inputs: {
        val expected: Output[Int]
      }

      val variables: {
        val estimatedDist: Variable[Float]
      }

      val outputs: {
        val choice: Output[Int]
        val loss: Output[Float]
      }
    }

    sealed abstract class Service(val session: core.client.Session,
                                  val model: Model)
        extends Recommender.Service

    def fresh(recommender: Recommender): Managed[Throwable, Recommender.Service] = {
      val impl = ZManaged.make(acquireFresh(recommender))(releaseFresh)
      impl.map(impl => impl: Recommender.Service)
    }

    def buildModel(recommender: Recommender, graph: Graph): Model = tf.createWith(graph) {
      import recommender.choices

      new Model {
        val inputs = new {
          val expected = Output.placeholder[Int](Shape(1))
        }

        val variables = new {
          val estimatedDist = tf.variable[Float]("estimatedDist",
                                                Shape(choices.size),
                                                tf.RandomUniformInitializer())
        }

        val outputs = new {
          val estimatedDist = variables.estimatedDist.toOutput

          val choice = tf.argmax(estimatedDist, 0, INT32)

          val loss: Output[Float] = {
            val expectedDist = tf.oneHot(inputs.expected, choices.size, 1f, 0f)
            tf.mean(tf.softmaxCrossEntropy(estimatedDist, expectedDist))
          }
        }
      }
    }

    def acquireFresh(recommender: Recommender): Task[Service] = Task {
      val graph = Graph()
      val model = buildModel(recommender, graph)
      val optimizer = ops.training.optimizers.GradientDescent(learningRate = 0.1f)
      val session = core.client.Session(graph)

      tf.createWith(graph) {
        session.run(targets = tf.globalVariablesInitializer())
      }

      new Service(session, model) {
        def recommend: Task[String] = Task {
          val predictedChoice: Int = session.run(fetches = model.outputs.choice).scalar
          recommender.choices(predictedChoice)
        }

        def update(actual: String): Task[Unit] =  Task {
          val expectedIndex = recommender.choices.indexOf(actual)
          tf.createWith(graph) {
            val updates = optimizer.minimize(model.outputs.loss)
            session.run(feeds = (model.inputs.expected -> (expectedIndex: Tensor[Int])),
                        targets = updates)
          }
        }
      }
    }

    def releaseFresh(service: Service) = UIO(service.session.close())
  }
}
