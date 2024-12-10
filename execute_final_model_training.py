from training.pipeline.final_train.feature_engineering import FeatureEngineeringTrainingPipeline
from training.pipeline.final_train.model_training import ModelTrainingPipeline
from training.pipeline.final_train.model_evaluation import ModelEvaluationTrainingPipeline
from training.custom_logging import info_logger

PIPELINE = "Feature Engineering Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
obj = FeatureEngineeringTrainingPipeline()
obj.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

PIPELINE = "Final Model Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
obj = ModelTrainingPipeline()
obj.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

PIPELINE = "Model Evaluation Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
obj = ModelEvaluationTrainingPipeline()
obj.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")