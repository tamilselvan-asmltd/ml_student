import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    dataset_path: str = '/Users/tamilselavans/Downloads/student_performance/src/dataset/dataset.csv'
    test_size: float = 0.2
    random_state: int = 42
    artifact_dir: str = os.path.join(os.getcwd(), 'artifact')
    train_data_path: str = os.path.join(artifact_dir, 'train_dataset.csv')
    test_data_path: str = os.path.join(artifact_dir, 'test_dataset.csv')

    def create_artifact_directory(self):
        # Create the artifact directory if it doesn't exist
        if not os.path.exists(self.artifact_dir):
            os.makedirs(self.artifact_dir)
            print(f"Directory {self.artifact_dir} created successfully!")
        else:
            print(f"Directory {self.artifact_dir} already exists.")

    def read_data(self):
        # Read the dataset from the specified path
        if os.path.exists(self.dataset_path):
            data = pd.read_csv(self.dataset_path)
            print(f"Data read successfully from {self.dataset_path}")
            return data
        else:
            raise FileNotFoundError(f"The file {self.dataset_path} does not exist.")

    def split_and_save_data(self, data):
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

        # Save the train and test data into the artifact directory
        train_data.to_csv(self.train_data_path, index=False)
        test_data.to_csv(self.test_data_path, index=False)
        print(f"Train dataset saved at {self.train_data_path}")
        print(f"Test dataset saved at {self.test_data_path}")

# Initialize data ingestion configuration
config = DataIngestionConfig()

# Create artifact directory if not exists
config.create_artifact_directory()

# Read data from the dataset
data = config.read_data()

# Split the data and save train/test datasets
config.split_and_save_data(data)
