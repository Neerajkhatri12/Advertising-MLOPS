import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

from training.custom_logging import info_logger, error_logger
from training.exception import CrossValError, handle_exception

from training.configuration_manager.configuration import ConfigurationManager
from training.entity.config_entity import CrossValConfig

class CrossVal:
    def __init__(self, config: CrossValConfig):
        self.config = config


    @staticmethod
    def is_json_serializable(value):
      """
      Check if a value is JSON serializable.
      """
      try:
          json.dumps(value)
          return True
      except (TypeError, OverflowError):
          return False
      
    def load_ingested_data(self):
        try:
            info_logger.info("Cross Validation Component started")
            info_logger.info("Loading ingested data")

            data_path = self.config.data_dir

            df = pd.read_csv(data_path, index_col=0)
            df.reset_index(drop=True, inplace=True)

            X = df.drop("sales", axis=1)
            y = df["sales"]

            info_logger.info("Ingested data loaded")
            return X, y

        except Exception as e:
            handle_exception(e, CrossValError)


    def split_data_for_final_train(self, X, y):
        try:
            info_logger.info("Data split for final train started")
            
            xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
            
            info_logger.info("Data split for final train completed")
            return xtrain, xtest, ytrain, ytest
        except Exception as e:
            handle_exception(e, CrossValError)
    
    def save_data_for_final_train(self, xtrain, xtest, ytrain, ytest):
        try:
            info_logger.info("Saving data for final train started")

            final_train_data_path = self.config.final_train_data_path
            final_test_data_path = self.config.final_test_data_path

            # Save xtrain and ytrain  to Train.npz
            # Save xtest and ytest to Test.npz
            np.savez(os.path.join(final_train_data_path, 'Train.npz'), xtrain=xtrain, ytrain=ytrain)
            np.savez(os.path.join(final_test_data_path, 'Test.npz'),  xtest=xtest, ytest=ytest)

            info_logger.info("Saved data for final train")
        except Exception as e:
            handle_exception(e, CrossValError)
    
    def run_cross_val(self, X, y):
        try:
            info_logger.info("Cross Validation Started")

            numeric_features = X.columns
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Missing value imputation
                ('scaler', StandardScaler())                 # Standardization
            ])

            # Step 3: Use ColumnTransformer to apply transformations
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                ]
            )

            # Step 4: Create a pipeline with Linear Regression
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])

            # Step 5: Set up GridSearchCV for hyperparameter tuning
            param_grid = {
                'regressor__fit_intercept': [True, False]  # Hyperparameter to tune
            }

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='r2',  # Use R-squared as the evaluation metric
                cv=5,  # Perform cross-validation on training data
                verbose=2
            )

            # Step 6: Fit GridSearchCV on the training data
            grid_search.fit(X, y)

            best_model = grid_search.best_estimator_
            best_params= grid_search.best_params_
            best_scores = grid_search.best_score_

            with open(self.config.STATUS_FILE, "a") as f:
                f.write(f"Best params for Model: {str(best_params)}\n")
                f.write(f"Best scoring(R2) for Model: {str(best_scores)}\n")

            best_model_params_path = os.path.join(self.config.best_model_params, f'best_params.json')
            best_model_params = best_model.get_params()
            serializable_params = {k: v for k, v in best_model_params.items() if self.is_json_serializable(v)}
            
            with open(best_model_params_path, 'w') as f:
                json.dump(serializable_params, f, indent=4)


            info_logger.info("Cross Validation completed")
        except Exception as e:
            handle_exception(e, CrossValError)

if __name__ == "__main__":
    config = ConfigurationManager()
    cross_val_config = config.get_cross_val_config()

    cross_val = CrossVal(config=cross_val_config)

    # Load the features and target
    X,y = cross_val.load_ingested_data()

    # Split the data into train and test sets for final train
    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X,y)

    # Save xtrain, xtest, ytain, ytest to be used final train
    cross_val.save_data_for_final_train(xtrain, xtest, ytrain, ytest)

    # Run cross validation
    cross_val.run_cross_val(xtrain, ytrain)
    