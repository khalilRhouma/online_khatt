import os
import numpy as np
from pathlib import Path
from random import shuffle
import shutil
from lang_trans.arabic import buckwalter

from src.features.extract_features import process_single_file
from src.features.alphabet import Alphabet


def create_points_and_target_inputs(data_path, save_dir):

    # create data input
    ink_paths = list(Path(data_path).glob("*_coor.txt"))

    os.makedirs(save_dir,exist_ok=True)
    print(len(ink_paths))
    for filename in ink_paths :
            with open(filename) as file:
                process_single_file(file,filename, f'{save_dir}/{filename.stem}_input.npy', normalize=True)
                file.close()
                try:
                    os.rename(filename,'processed/'+filename)
                except Exception as e:
                    pass

    # create target inputs
    # create buckwalter alphabet
    backwalter_labels = ["'", "|", ">", "&", "<", "}", "A", "b", "p", "t", "v", "j", "H", "x", "d", "*",
                    "r", "z", "s", "$", "S", "D", "T", "Z", "E", "g", "_", "f", "q", "k",  "l", "m", 
                        "n", "h", "w", "Y", "y", "F", "N", "K", "a", "u", "i",  "~", "o", "`", "{", "P",
                        "J", "V", "G", " ", ".", ':','=','«','»','(', ')','[',']','/','؟','!',"%",'0','1','2','3','4','4','5','6','7','8','9', "#\n"]
    with open("backwalter_labels.txt","w") as f:
        f.writelines("\n".join(backwalter_labels))

    alphabet=Alphabet('backwalter_labels.txt')
    text_target_path = list(Path(data_path).glob("*.txt"))

    os.makedirs(save_dir,exist_ok=True)
    for file_name in text_target_path:
        unicode = ['\ufeff','\xa0','،','–','‐','؛',',','"','؟','?','‘','\n']
        if "coor" not in file_name.name:
            with open(file_name,'r') as f:
                text = f.read()
                for i in unicode:
                    if i in text:
                        text= text.replace(i," ")
            label = buckwalter.transliterate(text.strip())
            # print(label)
            np.save(str(save_dir)+'/'+ file_name.name.replace('.txt','_target.npy'), alphabet.encode(label))

    # delete "_coor" from file name
    for file in os.listdir(save_dir):
        if "_coor" in file:
            os.rename(os.path.join(save_dir,file), os.path.join(save_dir,file.replace('_coor','')))

def create_test_set(val_save, test_save):
    
    val_files = os.listdir(val_save)
    print(len(val_files))
    val_files_names = list({f.rsplit('_',1)[0] for f in val_files})
    shuffle(val_files_names)
    partition = len(val_files_names) // 2 + 1
    val_files, test_files = val_files_names[:partition], val_files_names[partition:]
    os.makedirs(test_save,exist_ok=True)
    for file in test_files:
        shutil.move(os.path.join(val_save,file+"_input.npy"),os.path.join(test_save,file+"_input.npy"))
        shutil.move(os.path.join(val_save,file+"_target.npy"),os.path.join(test_save,file+"_target.npy"))

def main(data_path):

    os.makedirs(data_path,exist_ok=True)

    train_data_path = "data/Training/"
    val_data_path = "data/Validation/"
    train_save = data_path + "train"
    dev_save = data_path + "dev"
    test_save = data_path + "test"

    print("Creating features for train set...")
    create_points_and_target_inputs(train_data_path, train_save)
    print("Creating features for dev and test set...")
    create_points_and_target_inputs(val_data_path, dev_save)
    print("Generating test set from validation data...")
    create_test_set(dev_save, test_save)

    print("Done!")

if __name__=="__main__":
    data_path = "preprocessed_data/"
    main(data_path)