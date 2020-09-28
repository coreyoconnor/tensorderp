package glngn.tensorderp.minirec

import org.platanios.tensorflow.api._
import zio._

/**
  * Pick an ice cream flavor:
  *
  * - chocolate
  * - vanilla
  * - caramel
  * - more caramel
  *
  * choice [$recommended]: _
  */
object IceCreamMenu {
  type Flavor = String
  val choices: List[Flavor] = List("chocolate", "vanilla", "caramel", "more caramel")
  val recommender = new Recommender(choices)
}

class Recommender(val choices: List[String]) {
  def fresh: Managed[Throwable, Recommender.Service] = providers.tensorflow.fresh(this)

  def constant(i: Int): Managed[Throwable, Recommender.Service] =
    providers.constant(i, this)

  def random: Managed[Throwable, Recommender.Service] = ???
}

object Recommender {
  def recommend: ZIO[Has[Service], Throwable, String] =
    ZIO.service[Service].flatMap(_.recommend)

  def update(actual: String): ZIO[Has[Service], Throwable, Unit] =
    ZIO.service[Service].flatMap(_.update(actual))

  trait Service {
    def recommend: IO[Throwable, String]
    def update(actual: String): IO[Throwable, Unit]
  }
}

object providers {
  def constant(i: Int, recommender: Recommender): Managed[Throwable, Recommender.Service] =
    ZManaged.effect {
      new Recommender.Service {
        def recommend: IO[Throwable, String] = UIO {
          recommender.choices.drop(i).head
        }

        def update(actual: String): IO[Throwable, Unit] = UIO.unit
      }
    }

  object tensorflow {
    sealed abstract class Service(val session: core.client.Session)
        extends Recommender.Service

    def fresh(recommender: Recommender): Managed[Throwable, Recommender.Service] = {
      ZManaged.make(acquireFresh(recommender))(releaseFresh)
    }

    trait Model {
      val variables: {
        val estimatedDist: Variable[Float]
      }

      val outputs: {
        val estimatedDist: Output[Float]
        val choice: Output[Int]
      }
    }

    def buildModel(recommender: Recommender, graph: Graph): Model = tf.createWith(graph) {
      val choices = recommender.choices

      new Model {
        val variables = new {
          val estimatedDist = tf.variable[Float]("estimatedDist",
                                                 Shape(choices.size),
                                                 tf.RandomUniformInitializer())
        }

        val outputs = new {
          val estimatedDist = variables.estimatedDist.toOutput
          val choice = tf.argmax(estimatedDist, 0, INT32)
        }
      }
    }

    def acquireFresh(recommender: Recommender): Task[Service] = Task {
      val graph = Graph()
      val model = buildModel(recommender, graph)
      val session = core.client.Session(graph)

      tf.createWith(graph) {
        session.run(targets = tf.globalVariablesInitializer())
      }

      new Service(session) {
        def recommend: Task[String] = Task {
          val predictedChoice = session.run(fetches = model.outputs.choice)
          val choiceIndex = predictedChoice.scalar
          recommender.choices(choiceIndex)
        }

        def update(actual: String): Task[Unit] = Task {
          ()
        }
      }
    }

    def releaseFresh(service: Service): UIO[Unit] = UIO {
      service.session.close()
    }
  }
}

object ConsoleIceCreamMenu extends App {
  override def run(args: List[String]) = {
    val menu = for {
      _ <- console.putStrLn("Pick an ice cream flavor:")
      _ <- ZIO.foreach(IceCreamMenu.choices) { choice =>
        console.putStrLn(s"- ${choice}")
      }
      recommended <- Recommender.recommend
      selectedFlavor <- {
        console.putStr(s"choice [${recommended}]: ") *> console.getStrLn
      }
      actualFlavor <- if (selectedFlavor.isEmpty) {
        UIO(recommended)
      } else {
        Recommender.update(selectedFlavor) *> UIO(selectedFlavor)
      }
      _ <- console.putStrLn(s"\nUser selected ${actualFlavor}")
    } yield ()

    val recommenderLayer = IceCreamMenu.recommender.fresh.toLayer
    menu.provideCustomLayer(recommenderLayer).exitCode
  }
}
