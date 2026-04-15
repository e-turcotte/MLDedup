package essent

import essent.Graph.NodeID
import _root_.logger.LazyLogging

import scala.io.Source

case class ModuleFeatures(
  instanceCount: Int,
  moduleIRSize: Int,
  boundarySignalCount: Int,
  boundaryToInteriorRatio: Double,
  edgeCountWithin: Int,
  fractionDesignCovered: Double,
  originalIRSize: Int
) {
  def toArray: Array[Double] = Array(
    instanceCount.toDouble,
    moduleIRSize.toDouble,
    boundarySignalCount.toDouble,
    boundaryToInteriorRatio,
    edgeCountWithin.toDouble,
    fractionDesignCovered,
    originalIRSize.toDouble
  )
}

object MLRankModel extends LazyLogging {

  private val ResourcePath = "META-INF/ml-rank-coefficients.csv"
  private val NumFeatures = 7

  /** 
   * Load the coefficients from the resource CSV file.
   * The first row is the header, the second row is the coefficients.
   * @return The coefficients as an array of doubles.
   */
  def loadCoefficients(): Option[Array[Double]] = {
    Option(getClass.getClassLoader.getResourceAsStream(ResourcePath)).flatMap { in =>
      try {
        val lines = Source.fromInputStream(in).getLines().toSeq
        in.close()
        if (lines.size < 2) {
          logger.error(s"Coefficients file $ResourcePath must have a header row and a data row")
          None
        } else {
          val values = lines(1).split(",").map(_.trim.toDouble)
          if (values.length != NumFeatures + 1) {
            logger.error(s"Expected ${NumFeatures + 1} coefficients (intercept + $NumFeatures features), got ${values.length}")
            None
          } else {
            Some(values)
          }
        }
      } catch {
        case e: Throwable =>
          logger.error(s"Failed to parse $ResourcePath: ${e.getMessage}")
          None
      }
    }
  }

  /**
   * Predict the speedup of a module using the coefficients.
   * @param coeffs The coefficients as an array of doubles, derived from the loadCoefficients() method.
   * @param features The features of the module, derived from the computeFeatures() method.
   * @return The predicted speedup as a double.
   */
  def predict(coeffs: Array[Double], features: ModuleFeatures): Double = {
    val intercept = coeffs(0)
    val featureArray = features.toArray
    var score = intercept
    for (i <- featureArray.indices) {
      score += coeffs(i + 1) * featureArray(i)
    }
    score
  }

  /**
   * Compute features for a candidate module using pre-dedup graph structure.
   * @param modName The name of the module.
   * @param modInstInfo The module instance information, derived from the ModuleInstanceInfo class.
   * @param sg The statement graph, derived from the StatementGraph class.
   * @param originalIRSize The original IR size of the design.
   * @return The features of the module as a ModuleFeatures object.
   */
  def computeFeatures(
    modName: String,
    modInstInfo: ModuleInstanceInfo,
    sg: StatementGraph,
    originalIRSize: Int
  ): ModuleFeatures = {
    val instances = modInstInfo.internalModInstanceTable(modName)
    val instanceCount = instances.size
    val moduleIRSize = modInstInfo.internalModIRSize(modName)
    val fractionDesignCovered = (instanceCount * moduleIRSize).toDouble / originalIRSize

    val instanceNodeSet = modInstInfo.instInclusiveNodesTable(instances.head).toSet

    val edgeCountWithin = instanceNodeSet.toSeq.map { nid =>
      sg.outNeigh(nid).count(instanceNodeSet.contains)
    }.sum

    // A statement-graph node is "boundary" if it has an edge to/from outside the instance's
    // subgraph — proxy for cross-instance coupling cost before any dedup transform is applied.
    val boundarySignalCount = instanceNodeSet.toSeq.count { nid =>
      sg.outNeigh(nid).exists(!instanceNodeSet.contains(_)) ||
      sg.inNeigh(nid).exists(!instanceNodeSet.contains(_))
    }

    val boundaryToInteriorRatio = boundarySignalCount.toDouble / originalIRSize

    ModuleFeatures(
      instanceCount = instanceCount,
      moduleIRSize = moduleIRSize,
      boundarySignalCount = boundarySignalCount,
      boundaryToInteriorRatio = boundaryToInteriorRatio,
      edgeCountWithin = edgeCountWithin,
      fractionDesignCovered = fractionDesignCovered,
      originalIRSize = originalIRSize
    )
  }

  /**
   * Score all candidate modules and return (bestModuleName, pseudoRank).
   * pseudoRank is the 1-based position of the selected module in the
   * original benefit-sorted list.
   * @param candidates The candidate modules to score.
   * @param modInstInfo The module instance information, derived from the ModuleInstanceInfo class.
   * @param sg The statement graph, derived from the StatementGraph class.
   * @param originalIRSize The original IR size of the design.
   * @param coeffs The coefficients as an array of doubles, derived from the loadCoefficients() method.
   * @param benefitSortedNames The original benefit-sorted list of module names.
   * @return The best module name and its pseudo-rank.
   */
  def selectBestModule(
    candidates: Seq[String],
    modInstInfo: ModuleInstanceInfo,
    sg: StatementGraph,
    originalIRSize: Int,
    coeffs: Array[Double],
    benefitSortedNames: Seq[String]
  ): (String, Int) = {
    val scored = candidates.map { modName =>
      val features = computeFeatures(modName, modInstInfo, sg, originalIRSize)
      val score = predict(coeffs, features)
      logger.info(f"ML score for [$modName]: $score%.6f  (instances=${features.instanceCount}, irSize=${features.moduleIRSize}, boundary=${features.boundarySignalCount}, coverage=${features.fractionDesignCovered}%.4f)")
      (modName, score)
    }

    val (bestMod, bestScore) = scored.maxBy(_._2)
    val pseudoRank = benefitSortedNames.indexOf(bestMod) + 1
    logger.info(f"ML model selected [$bestMod] with predicted speedup $bestScore%.6f (pseudo-rank $pseudoRank)")
    (bestMod, pseudoRank)
  }
}
