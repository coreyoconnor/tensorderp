package glngn.tensorderp.minirec.draft0

import org.platanios.tensorflow.api._
import zio._

/**
  * Pick one:
  * 1) choco
  * 2) vanilla
  * 3) caramel
  * 4) more caramel
  *
  * choice [$recommendation]: ?
  */
object IceCreamMenu {
  type Flavor = String
  val choices: List[Flavor] = List("chocolate", "vanilla", "caramel", "more caramel")
  val recommender = new Recommender(choices)
}

object IceCreamMenuEx0 {
  def main(args: Array[String]): Unit = {
    val menu = for {
      _ <- ZIO.foreach(IceCreamMenu.choices) { choice => console.putStrLn(s"- ${choice}") }
      recommendedFlavor <- Recommender.recommend
      _ <- console.putStrLn(s"choice [$recommendedFlavor]: ")
      _ <- Recommender.update("caramel")
    } yield()

    val io = menu.provideCustomLayer(IceCreamMenu.recommender.fresh)

    Runtime.default.unsafeRun(io)
  }
}

object Recommender {
  def recommend: ZIO[Has[Service], Throwable, String] =
    ZIO.service[Recommender.Service].flatMap(_.recommend)

  def update(actual: String): ZIO[Has[Service], Throwable, Unit] =
    ZIO.service[Recommender.Service].flatMap(_.update(actual))

  def impl: ZIO[Has[Service], Nothing, Impl] =
    ZIO.service[Recommender.Service].map(_.impl)

  trait Service {
    def recommend: IO[Throwable, String]
    def update(actual: String): IO[Throwable, Unit]
    val impl: Impl
  }

  trait Impl {
    val session: Session
    val graph: Graph

    val inputs: {}

    val variables: {
      val estimatedDist: Variable[Float]
    }

    val outputs: {
      val choice: Output[Int]
      def loss(expected: Int): Output[Float]
    }
  }
}

class Recommender(choices: Seq[String]) {
  def fresh: Layer[Throwable, Has[Recommender.Service]] =
    ZLayer.fromAcquireRelease(acquireFreshSession)(releaseSession)

  def acquireImpl: Task[Recommender.Impl] = Task {
    tf.createWith(Graph()) {
      new Recommender.Impl {
        object inputs

        val variables = new {
          val estimatedDist = tf.variable[Float]("estimatedDist",
                                                 Shape(choices.size),
                                                 tf.RandomUniformInitializer())
        }

        val outputs = new {
          val estimatedDist = variables.estimatedDist.toOutput

          val choice = tf.argmax(estimatedDist, 0, INT32)

          def loss(expected: Int): Output[Float] = {
            val expectedDist = tf.oneHot(expected, choices.size, 1f, 0f)
            tf.mean(tf.softmaxCrossEntropy(estimatedDist, expectedDist))
          }
        }

        val graph = tf.currentGraph
        val session = core.client.Session(graph)
        session.run(targets = tf.globalVariablesInitializer())
      }
    }
  }

  def releaseImpl: Recommender.Impl => UIO[_] = { impl =>
    UIO(impl.session.close)
  }

  val acquireFreshSession: Task[Recommender.Service] = for {
    acquiredImpl <- acquireImpl
    optimizer <- Task(ops.training.optimizers.GradientDescent(learningRate = 0.1f))
  } yield new Recommender.Service {
    val impl = acquiredImpl

    def recommend: Task[String] = Task {
      val predictedChoice: Int = impl.session.run(fetches = impl.outputs.choice).scalar
      choices(predictedChoice)
    }

    def update(actual: String): Task[Unit] =  Task {
      val expectedIndex = choices.indexOf(actual)
      val loss = impl.outputs.loss(expectedIndex)
      val updates = optimizer.minimize(loss)
      impl.session.run(targets = updates)
    }
  }

  def releaseSession(recommender: Recommender.Service) = releaseImpl(recommender.impl)
}
