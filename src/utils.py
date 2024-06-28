import pickle
import os
from sklearn.metrics import r2_score

def save_file_as_pickle(name,path):
    dir_name=os.path.dirname(path)
    os.makedirs(dir_name,exist_ok=True)
    with open(path,"wb") as file_path:
        pickle.dump(name,file_path)


def evaluate_model(true_value,predicted_value):
    r2=r2_score(true_value,predicted_value)
    return r2

def load_model(path):
    with open(path,"rb") as file:
       return pickle.load(file)