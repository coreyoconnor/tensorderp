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
  * choice: _
  */
object IceCreamMenu {
  type Flavor = String
  val choices: List[Flavor] = List("chocolate", "vanilla", "caramel", "more caramel")
}

object ConsoleIceCreamMenu extends App {
  override def run(args: List[String]) = {
    val menu = for {
      _ <- console.putStrLn("Pick an ice cream flavor:")
      _ <- ZIO.foreach(IceCreamMenu.choices) { choice =>
        console.putStrLn(s"- ${choice}")
      }
      selectedFlavor <- console.putStr(s"choice: ") *> console.getStrLn
      _ <- console.putStrLn(s"\nUser selected ${selectedFlavor}")
    } yield ()

    menu.exitCode
  }
}
