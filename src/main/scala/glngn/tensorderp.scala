package glngn.tensorderp

import org.platanios.tensorflow.api.{ops => OOps, _}
import tensors.{ ops => TOps }
import de.sciss.synth.{ io => audioIO }
import java.lang.Math

final object TestAmplitudes {
  val allZeros = Stream.fill[Float](512)(0f)
  val sineWave1CyclePer512 = (0 until 512).toStream map { i =>
    val r: Double = (i.toDouble / 512.0) * 2.0 * Math.PI
    Math.sin(r).toFloat
  }
  val indices512 = (0 until 512).toSeq
}

// tensor is ordered oldest to newest left to right.
// at 0 is the oldest
// at -1 is the newest
class TensorAudioIterator(amplitudes: Stream[Float], size: Int = 512) {
  val initial = Tensor.zeros[Float](Shape(size))
  private var prev: Option[Tensor[Float]] = None
  private val nextAmplitude: Iterator[Float] = amplitudes.toIterator

  def next: Option[Tensor[Float]] = {
    val nextPrev = prev match {
      case _ if !nextAmplitude.hasNext => None
      case None => Some(initial)
      case Some(prevT) => {
        var sample = nextAmplitude.next
        // eh, close enough
        if (sample <= -1.0)
          sample = -0.999999f
        if (sample >= 1.0)
          sample = 0.999999f

        val out = TOps.Basic.concatenate(Seq(prevT(1 ::), Tensor(sample)), 0)
        Some(out)
      }
    }
    prev = nextPrev
    prev
  }
}

final object TestAudioIterators {
  def sine = new TensorAudioIterator(TestAmplitudes.sineWave1CyclePer512, 512)
}

class QuantizedAudioTransforms(quantizedSize: Int = 255) {
  val quantizedSizeT = quantizedSize.toFloat: Tensor[Float]
  val quantizedSizeO = quantizedSize.toFloat: Output[Float]
  val quantizedSizeLog1pT = {
    import TOps.Math.log1p
    log1p(quantizedSizeT)
  }
  val quantizedSizeLog1pO = {
    import OOps.Math.log1p
    log1p(quantizedSizeO)
  }

  // in is (-1.0, 1.0)
  // out is (-1.0, 1.0)
  def muLawCompandingT(v: Tensor[Float]): Tensor[Float] = {
    import TOps.Math.{abs, log1p, sign}

    sign(v) * log1p(abs(v) * quantizedSizeT) / quantizedSizeLog1pT
  }

  def muLawCompandingInverseT(v: Tensor[Float]): Tensor[Float] = {
    import TOps.Math.{abs, expm1, sign}

    // x = sign(v) * log1p(abs(v) * quantizedSizeT) / quantizedSizeLog1pT
    // expm1(abs(x) * quantizedSizeLog1pT)*sign(x) = v
    sign(v) * expm1(abs(v) * quantizedSizeLog1pT) / quantizedSizeT
  }

  def muLawCompandingO(v: Output[Float]): Output[Float] = {
    import OOps.Math.{abs, log1p, sign}

    sign(v) * log1p(abs(v) * quantizedSizeO) / quantizedSizeLog1pO
  }

  // in is (-1.0, 1.0)
  // out is [0, 256)
  def quantizedT(v: Tensor[Float]): Tensor[Int] = {
    TOps.Math.floor(v * 128.0f + 127.5f).toInt
  }

  def quantizedOneHotT(v: Tensor[Int]): Tensor[Float] = {
    Tensor.oneHot(v, 256, 1.0f, 0.0f)
  }

  def input(v: Tensor[Float]): Tensor[Float] = {
    quantizedOneHotT(quantizedT(muLawCompandingT(v)))
  }

  def deQuantizedOneHotT(v: Tensor[Float]): Tensor[Long] = {
    TOps.Math.argmax(v, -1)
  }

  def deQuantizedT(v: Tensor[Long]): Tensor[Float] = {
    (v.toFloat - 127.5f)/128f
  }

  def output(v: Tensor[Float]): Tensor[Float] = {
    muLawCompandingInverseT(deQuantizedT(deQuantizedOneHotT(v)))
  }
}

final object Transform {
  val audioTransforms = new QuantizedAudioTransforms()
  import audioTransforms._

  def muLawCompandingTable = new {
    val x = tfi.range(-1.0f, 1.0f, 0.01f)
    val y = muLawCompandingT(x)
    val z = muLawCompandingInverseT(y)
  }

  def quantizedTable = new {
    val x = muLawCompandingTable.x
    val y = quantizedT(muLawCompandingTable.y)
  }

  val testAmplitudes = new {
    val a0 = 0.0f
    val a1 = 1.0f
    val a2 = -1.0f
    val a3 = 0.5f
    val a4 = -0.5f
    val all = Tensor(a0, a1, a2, a3, a4)
  }

  def muLawCompandingTestOut: Tensor[Float] = {
    muLawCompandingT(testAmplitudes.all)
  }

  def oneHotTestOut: Tensor[Float] = {
    quantizedOneHotT(quantizedT(muLawCompandingTestOut))
  }

  val muLawTestIn = Output.placeholder[Float](name = "normalizedAmplitude",
                                              shape = Shape(1))
  def muLawTestGraph = {
    muLawCompandingO(muLawTestIn)
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
