from src.components.data_ingestion import InitiateDataIngestion


if __name__ == "__main__":
    di=InitiateDataIngestion()
    train_dataset_path,test_dataset_path=di.initiate_data_ingestion()
    print(train_dataset_path,test_dataset_path)