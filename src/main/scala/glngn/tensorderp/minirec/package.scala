package glngn.tensorderp.minirec

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

class Recommender(choices: List[String])

object IceCreamMenuEx0 {
  def main(args: Array[String]): Unit = {
    val menu = for {
      _ <- ZIO.foreach(IceCreamMenu.choices) { choice => console.putStrLn(s"- ${choice}") }
      selectedFlavor <- (console.putStr(s"choice: ") *> console.getStrLn).repeatUntil(_.trim.nonEmpty)
      _ <- console.putStrLn(s"\nUser selected ${selectedFlavor}")
    } yield ()

    val io = menu

    Runtime.default.unsafeRun(io)
  }
}
