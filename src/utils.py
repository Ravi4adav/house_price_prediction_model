import os
import pickle


def save_object(file_path, file_name, file_obj):
    path=os.makedirs(file_path,exist_ok=True)
    file=os.path.join(file_path, file_name)
    print(f"Opening file: {file}")
    with open(file,'wb') as preprocessor_file:
        pickle.dump(file_obj,preprocessor_file)
    return None


def load_object(preprocessor_file_path):
    with open(preprocessor_file_path,'rb') as file:
        preprocessor=pickle.load(file)
    return preprocessor


