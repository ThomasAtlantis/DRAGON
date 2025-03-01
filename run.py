import sys; sys.path.append(".")

# Language Modeling
# from experiments.LanguageModeling.eval import (
#     LanguageModelingEvaluator as Evaluator, 
#     LanguageModelingConfig as Config
# )

# RagSequenceForGeneration vs. RagTokenForGeneration
from experiments.SeqAggVsTokAgg.eval import (
    SeqAggVsTokAggEvaluator as Evaluator,
    SeqAggVsTokAggConfig as Config
)


if __name__ == "__main__":
    config = Config()
    config.parse_sys_args()
    evaluator = Evaluator(config)
    evaluator.evaluate()