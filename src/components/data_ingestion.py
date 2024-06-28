import pandas as pd
from sklearn.model_selection import train_test_split
import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_dataset_path=os.path.join("artifacts/traintestdataset","train_dataset.csv")
    test_dataset_path=os.path.join("artifacts/traintestdataset","test_dataset.csv")

class InitiateDataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv("dataset/student_performance_indicator.csv")
            train_dataset,test_dataset=train_test_split(df,test_size=0.3,random_state=69)
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_dataset_path),exist_ok=True)
            train_dataset.to_csv(self.data_ingestion_config.train_dataset_path,header=True,index=False)
            test_dataset.to_csv(self.data_ingestion_config.test_dataset_path,header=True,index=False)
            return (self.data_ingestion_config.train_dataset_path,self.data_ingestion_config.test_dataset_path)
        except Exception as e:
            raise e


