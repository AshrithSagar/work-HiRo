"""
Test DemInf
"""
# tests/test_deminf.py

from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer import console
from pacer.deminf import DemInfEstimator, DemInfScorer
from pacer.testutils import get_demonstrations

set_global_default_runtime_options(RuntimeOptions(validate=True))


if __name__ == "__main__":
    demonstrations = get_demonstrations(choice="FROM_LASA", pattern="GShape")

    estimator = DemInfEstimator(k=3)
    scorer = DemInfScorer(estimator)

    scores = scorer.score_demonstrations(demonstrations)
    rankings = scorer.rank_scores(scores)
    console.print("rankings=", rankings)
