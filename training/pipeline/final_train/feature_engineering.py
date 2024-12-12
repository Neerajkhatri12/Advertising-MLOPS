from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.feature_engineering import FeatureEngineering
from training.custom_logging import info_logger
import sys

PIPELINE = "Feature Engineering Training Pipeline"

class FeatureEngineeringTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()

        feature_engineering = FeatureEngineering(config = feature_engineering_config)
        xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()
        xtrain, xtest, ytrain, ytest = feature_engineering.transform_data(xtrain, xtest, ytrain, ytest)
        feature_engineering.save_transformed_data(xtrain, xtest, ytrain, ytest)   

        
                       
if __name__ == "__main__":
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    obj = FeatureEngineeringTrainingPipeline()
    obj.main()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")