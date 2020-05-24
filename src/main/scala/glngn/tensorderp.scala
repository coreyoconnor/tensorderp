package glngn.tensorderp

import org.platanios.tensorflow.api.{ops => OOps, _}
import tensors.{ ops => TOps }
import de.sciss.synth.{ io => audioIO }
import java.lang.{Math => JavaMath}

final object TestAmplitudes {
  val frameSize = 512

  val allZeros = Stream.fill[Float](frameSize)(0f)

  val sineWave1CyclePer512 = (0 until frameSize).toStream map { i =>
    val r: Double = (i.toDouble / frameSize.toFloat) * 2.0 * JavaMath.PI
    JavaMath.sin(r).toFloat
  }

  val indices512 = (0 until frameSize).toSeq
}

// tensor is ordered oldest to newest left to right.
// at 0 is the oldest
// at -1 is the newest
class TensorAudioIterator(amplitudes: Stream[Float], frameSize: Int = 512) {
  val zeros = Tensor.zeros[Float](Shape(frameSize))

  sealed trait State
  object State {
    case object Initial extends State
    case class Forwarding(prior: Tensor[Float]) extends State
    case class Tail(prior: Tensor[Float], remaining: Int = 0) extends State
    case object Done extends State
  }
  private var state: State = State.Initial
  private val nextAmplitude: Iterator[Float] = amplitudes.toIterator

  def sample: Float = {
    var out = nextAmplitude.next
    // eh, close enough
    if (out <= -1.0f)
      out = -1.0f
    if (out >= 1.0f)
      out = 1.0f
    out
  }

  def next: Option[Tensor[Float]] = {
    val (out, nextState) = state match {
      case State.Initial => {
        (Some(zeros), State.Forwarding(zeros))
      }

      case State.Forwarding(prior) =>
        if (nextAmplitude.hasNext) {
          val out = TOps.Basic.concatenate(Seq(prior(1 ::), Tensor(sample)), 0)
          (Some(out), State.Forwarding(out))
        } else {
          //val out = TOps.Basic.concatenate(Seq(prior(1 ::), Tensor(0.0f)), 0)
          //(Some(out), State.Tail(out))
          (None, State.Done)
        }

      case State.Tail(prior, 0) => (None, State.Done)

      case State.Tail(prior, n) => {
        val out = TOps.Basic.concatenate(Seq(prior(1 ::), Tensor(0.0f)), 0)
        (Some(out), State.Tail(out, n - 1))
      }
      case State.Done => (None, State.Done)
    }
    state = nextState
    out
  }

  def batchedNext(batchSize: Int): Option[Seq[Tensor[Float]]] = {
    val out = collection.mutable.Buffer.empty[Tensor[Float]]
    for(i <- 0 until batchSize) {
      next match {
        case None => ()
        case Some(t) => out += t
      }
    }
    if (out.size != batchSize) None else Some(out.toSeq)
  }
}

final object TestAudioIterators {
  def sine = new TensorAudioIterator(TestAmplitudes.sineWave1CyclePer512, 512)
}

class QuantizedAudioTransforms(val quantizedSize: Int = 256) {
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

  def muLawCompandingInverseO(v: Output[Float]): Output[Float] = {
    import OOps.Math.{abs, expm1, sign}

    // x = sign(v) * log1p(abs(v) * quantizedSizeT) / quantizedSizeLog1pT
    // expm1(abs(x) * quantizedSizeLog1pT)*sign(x) = v
    sign(v) * expm1(abs(v) * quantizedSizeLog1pT) / quantizedSizeT
  }

  // in is (-1.0, 1.0)
  // out is [0, 256)
  def quantizedT(v: Tensor[Float]): Tensor[Int] = {
    TOps.Math.floor(v * (quantizedSize.toFloat / 2f - 0.5f) + (quantizedSize.toFloat / 2f)).toInt
  }

  def quantizedO(v: Output[Float]): Output[Int] = {
    OOps.Math.floor(v * (quantizedSize.toFloat / 2f - 0.5f) + (quantizedSize.toFloat / 2f)).toInt
  }

  def quantizedOneHotT(v: Tensor[Int]): Tensor[Float] = {
    Tensor.oneHot(v, quantizedSize, 1.0f, 0.0f)
  }

  def quantizedOneHotO(v: Output[Int]): Output[Float] = {
    OOps.Basic.oneHot(v, quantizedSize, 1.0f, 0.0f)
  }

  def inputT(v: Tensor[Float]): Tensor[Float] = {
    quantizedOneHotT(quantizedT(muLawCompandingT(v)))
  }

  def inputO(v: Output[Float]): Output[Float] = {
    quantizedOneHotO(quantizedO(muLawCompandingO(v)))
  }

  def deQuantizedOneHotT(v: Tensor[Float]): Tensor[Long] = {
    TOps.Math.argmax(v, -1)
  }

  def deQuantizedOneHotO(v: Output[Float]): Output[Long] = {
    OOps.Math.argmax(v, -1, INT64)
  }

  def deQuantizedT(v: Tensor[Long]): Tensor[Float] = {
    (v.toFloat - (quantizedSize.toFloat / 2f - 0.5f))/(quantizedSize.toFloat / 2f)
  }

  def deQuantizedO(v: Output[Long]): Output[Float] = {
    (v.toFloat - (quantizedSize.toFloat / 2f - 0.5f))/(quantizedSize.toFloat / 2f)
  }

  def outputT(v: Tensor[Float]): Tensor[Float] = {
    muLawCompandingInverseT(deQuantizedT(deQuantizedOneHotT(v)))
  }

  def outputO(v: Output[Float]): Output[Float] = {
    muLawCompandingInverseO(deQuantizedO(deQuantizedOneHotO(v)))
  }
}

class ConcatDense(frameSize: Int = 512, quantizedSize: Int = 256) {
  import OOps._

  val weights = tf.variable[Float]("output-weights",
                                    Shape(quantizedSize, frameSize*quantizedSize),
                                    tf.ZerosInitializer)

  def flatten(v: Output[Float]): Output[Float] =
    Basic.reshape(v, Tensor(-1, frameSize * quantizedSize))

  def op(v: Output[Float]): Output[Float] = {
    val flattened = flatten(v)
    Math.tensorDot(flattened, weights, Seq(-1), Seq(1))
  }
}

case class TrainResult(out: Tensor[Float],
                       loss: Tensor[Float])

class IteratedTrainSession(batchSize: Int = 64, frameSize: Int = 512, quantizedSize: Int = 256) {
  // immutable
  val audioTransforms = new QuantizedAudioTransforms(quantizedSize)
  val outLayer = new ConcatDense(frameSize, quantizedSize)

  // mutated by side effects
  var audioIterator = new TensorAudioIterator(TestAmplitudes.sineWave1CyclePer512, frameSize)
  val session = core.client.Session()

  val inWaveform = Output.placeholder[Float](name = "input-waveform",
                                             shape = Shape(batchSize, frameSize, quantizedSize))
  val actualWaveform = Output.placeholder[Float](name = "actual-waveform",
                                                 shape = Shape(batchSize, frameSize, quantizedSize))
  val outputSample = outLayer.op(inWaveform)

  val loss = tf.mean(tf.softmaxCrossEntropy(outputSample, actualWaveform(::, -1, ::)))
  val optimizer = tf.train.AdaGrad(0.01f).minimize(loss)

  val fetches = Seq(outputSample, loss)

  session.run(targets = tf.globalVariablesInitializer())

  def step: TrainResult = {
    val next = audioIterator.batchedNext(batchSize) match {
      case None => {
        audioIterator = new TensorAudioIterator(TestAmplitudes.sineWave1CyclePer512, frameSize)
        audioIterator.batchedNext(batchSize).get
      }
      case Some(next) => next
    }

    val feeds = Map(
      inWaveform -> audioTransforms.inputT(next: Tensor[Float]),
      actualWaveform -> audioTransforms.inputT(next: Tensor[Float])
    )

    val Seq(nextOut, currentLoss) = session.run(feeds = feeds, fetches = Seq(outputSample, loss), targets = optimizer)
    TrainResult(audioTransforms.outputT(nextOut), currentLoss)
  }
}

final object Main {
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

  def sineIdentityTest = {
    val transformedInput = inputT(TestAmplitudes.sineWave1CyclePer512)
    outputT(transformedInput)
  }

  val testAmplitudes = new {
    val a0 = 0.0f
    val a1 = 1.0f
    val a2 = -1.0f
    val a3 = 0.5f
    val a4 = -0.5f
    val all = Tensor(a0, a1, a2, a3, a4)
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

  def sineIdentityRun = {
    val in = Output.placeholder[Float](name = "waveform",
                                       shape = Shape(512))
    val feeds = in -> (TestAmplitudes.sineWave1CyclePer512: Tensor[Float])

    val graph = {
      outputO(inputO(in))
    }

    val session = core.client.Session()
    session.run(feeds = feeds, fetches = graph)
  }

  def singleLayerSineIdentityRun = {
    val frameSize = TestAmplitudes.frameSize
    val in = Output.placeholder[Float](name = "waveform",
                                       shape = Shape(2, frameSize))
    val feeds = in -> Tensor(TestAmplitudes.allZeros, TestAmplitudes.sineWave1CyclePer512: Tensor[Float])

    val graph = {
      val input = inputO(in)
      val outLayer = new ConcatDense(frameSize, quantizedSize)
      outputO(outLayer.op(input))
    }

    val session = core.client.Session()
    session.run(targets = tf.globalVariablesInitializer())
    session.run(feeds = feeds, fetches = graph)
  }
}
