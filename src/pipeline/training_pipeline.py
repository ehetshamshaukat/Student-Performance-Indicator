from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


if __name__ == "__main__":
    di=DataIngestion()
    train_dataset_path,test_dataset_path=di.initiate_data_ingestion()
    print(train_dataset_path,test_dataset_path)
    dt=DataTransformation()
    transformed_train_data,transformed_test_data=dt.initiate_data_transformation(train_dataset_path,test_dataset_path)