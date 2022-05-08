resolvers += Resolver.sonatypeRepo("snapshots")
scalaVersion := "2.13.6"

val zio = Seq(
  "dev.zio" %% "zio" % Versions.zio
)

val tensorflow = Seq(
  "org.platanios" %% "tensorflow" % Versions.tensorflow classifier "linux",
  "org.platanios" %% "tensorflow-data" % Versions.tensorflow
)

val audiofile = Seq(
  "de.sciss" %% "audiofile" % Versions.audiofile
)

lazy val root = (project in file("."))
  .settings(
    name := "tensorderp",
    organization := "glngn",
    version := "0.2.0-SNAPSHOT",
    libraryDependencies ++= tensorflow ++ audiofile ++ zio,
    libraryDependencies += "com.chuusai" %% "shapeless" % Versions.shapeless
  )
