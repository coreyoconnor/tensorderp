val zio = Seq(
  "dev.zio" %% "zio" % "1.0.0"
)

val tensorflow = Seq(
  "org.platanios" %% "tensorflow" % "0.4.1" classifier "linux-cpu-x86_64",
  "org.platanios" %% "tensorflow-data" % "0.4.1"
)

val audiofile = Seq(
  "de.sciss" %% "audiofile" % "1.5.1"
)

lazy val root = (project in file("."))
  .settings(
    name := "tensorderp",
    organization := "glngn",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := "2.12.10",
    libraryDependencies ++= tensorflow ++ audiofile ++ zio,
    libraryDependencies += "com.chuusai" %% "shapeless" % "2.4.0-M1"
  )
