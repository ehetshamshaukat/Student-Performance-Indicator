import os
from dataclasses import dataclass
from src.utils import save_file_as_pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    data_transformation_pickle_path=os.path.join("artifacts/pickle","data_transformation.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def data_transformation(self):
        try:

            numerical_columns=["reading_score","writing_score"]
            categorical_columns_ordinal=["parental_level_of_education","race_ethnicity"]
            categorical_columns_one_hot=["gender","lunch","test_preparation_course"]

            parental_level_of_education_ranking=['some high school',"high school","associate's degree",'some college',"bachelor's degree","master's degree"]
            race_ethnicity_ranking=["group A","group B","group C","group D","group E"]

            numerical_column_pipeline=Pipeline(steps=[
                ("impute",SimpleImputer(strategy="median")),
                ("standardscaler",StandardScaler())
            ])
            categorical_column_pipeline_ordinal=Pipeline(steps=[
                ("impute",SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder",OrdinalEncoder(categories=[parental_level_of_education_ranking,race_ethnicity_ranking])),
                ("standardscaler",StandardScaler())
            ])
            categorical_column_pipeline_one_hot=Pipeline(steps=[
                ('impute',SimpleImputer(strategy="most_frequent")),
                ("onehotencoder",OneHotEncoder(sparse_output=False)),
                ("standardscaler",StandardScaler())
            ])
            processing=ColumnTransformer([
                ("numerical_columns",numerical_column_pipeline,numerical_columns),
                ("categorical_columns_ordinal",categorical_column_pipeline_ordinal,categorical_columns_ordinal),
                ("categorical_columns_one_hot",categorical_column_pipeline_one_hot,categorical_columns_one_hot)
            ])
            return processing
        except Exception as e:
            raise e

    def initiate_data_transformation(self,train_path,test_path):
        try:

            train_dataset = pd.read_csv(train_path)
            test_dataset=pd.read_csv(test_path)

            dt=self.data_transformation()

            target_column="math_score"
            column_to_drop="math_score"

            xtrain=train_dataset.drop(columns=column_to_drop,axis=1)
            ytrain=train_dataset[target_column]

            xtest=test_dataset.drop(columns=column_to_drop,axis=1)
            ytest=test_dataset[target_column]

            xtrain_transformed_data=dt.fit_transform(xtrain)
            xtest_transformed_data=dt.transform(xtest)

            transformed_train_data=np.c_[xtrain_transformed_data,np.array(ytrain)]
            transformed_test_data=np.c_[xtest_transformed_data,np.array(ytest)]

            save_file_as_pickle(dt,self.data_transformation_config.data_transformation_pickle_path)
            return (transformed_train_data,transformed_test_data)

        except Exception as e:
            raise e