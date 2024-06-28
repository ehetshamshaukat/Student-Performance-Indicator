import pandas as pd
import os

from src.utils import load_model


class Prediction:
    def __init__(self):
        pass
    def initiate_prediction(self,feature):
        data_transformation_path=os.path.join("artifacts/pickle","data_transformation.pkl")
        model_path=os.path.join("artifacts/pickle","model.pkl")

        data_transformation=load_model(data_transformation_path)
        model=load_model(model_path)

        transformed_data=data_transformation.transform(feature)
        op=model.predict(transformed_data)
        return op

class GetFeatures:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    def to_dataframe(self):
        features_as_dataframe={
            "gender":[self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test_preparation_course":[self.test_preparation_course],
            "writing_score":[self.writing_score],
            "reading_score":[self.reading_score]
        }
        features=pd.DataFrame(features_as_dataframe)
        return features

