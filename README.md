# tensorderp

- https://arxiv.org/pdf/1609.03499.pdf

softmax rescale is

sign(x)*Math.log1p(255.0 * x)/Math.log1p(255.0)

```
interp.repositories() ++= Seq(coursierapi.IvyRepository.of(
    "file:///home/coconnor/.ivy2/local/[defaultPattern]"
))
interp.load.ivy(
    "glngn" %% "tensorderp" % "0.1.0-SNAPSHOT",
    ("org.platanios" %% "tensorflow" % "0.4.1").withClassifier("linux-cpu-x86_64")
)
```
