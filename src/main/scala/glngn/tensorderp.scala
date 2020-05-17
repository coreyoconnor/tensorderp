package glngn

import org.platanios.tensorflow.api._

package object tensorderp {
  val quantizedSize = 255
  val quantizedSizeT = quantizedSize.toFloat: Tensor[Float]
  val quantizedSizeO = quantizedSize.toFloat: Output[Float]
  val quantizedSizeLog1pT = {
    import tensors.ops.Math.log1p
    log1p(quantizedSizeT)
  }
  val quantizedSizeLog1pO = {
    import ops.Math.log1p
    log1p(quantizedSizeO)
  }

  def muLawCompandingTransformT(v: Tensor[Float]): Tensor[Float] = {
    import tensors.ops.Math.{abs, log1p, sign}

    sign(v) * log1p(abs(v) * quantizedSizeT) / quantizedSizeLog1pT
  }

  def muLawCompandingTable = new {
    val x = tfi.range(-1.0f, 1.0f, 0.01f)
    val y = muLawCompandingTransformT(x)
  }

  def muLawCompandingTransformO(v: Output[Float]): Output[Float] = {
    import ops.Math.{abs, log1p, sign}

    sign(v) * log1p(abs(v) * quantizedSizeO) / quantizedSizeLog1pO
  }

  // input is [-32768, 32787]
  def normalizeSamples(v: Output[Short]): Output[Float] = {
    v.toFloat / 32768.0f
  }

  // input is [-1.0, 1.0]
  // output is [-32768, 32767]
  // not exactly the reverse of normalize :\
  def deNormalizeSamples(v: Output[Float]): Output[Short] = {
    import ops.Clip.clipByValue

    clipByValue(v * 32768.0f, Short.MinValue.toFloat, Short.MaxValue.toFloat).toShort
  }

  val testAmplitudes = new {
    val a0 = 0.0f
    val a1 = 1.0f
    val a2 = -1.0f
    val a3 = 0.5f
    val a4 = -0.5f
  }

  def muLawCompandingTestOut: List[Tensor[Float]] = {
    List(
      testAmplitudes.a0,
      testAmplitudes.a1,
      testAmplitudes.a2,
      testAmplitudes.a3,
      testAmplitudes.a4
    ).map(muLawCompandingTransformT(_))
  }

  val muLawTestIn = Output.placeholder[Float](name = "normalizedAmplitude",
                                              shape = Shape(1))
  def muLawTestGraph = {
    muLawCompandingTransformO(muLawTestIn)
  }

  def muLawTestFeeds = {
    muLawTestIn -> (0.5f: Tensor[Float])
  }

  def muLawTestFetches = {
    Seq(muLawTestGraph)
  }

  def muLawTestRun = {
    val session = core.client.Session()
    val List(out) = session.run(feeds = muLawTestFeeds, fetches = muLawTestFetches)
    out
  }
}
