
from xml.etree import ElementTree as ET
import os
from tqdm import tqdm


def convert_inkml_to_points(data_path: str, data_type="train") -> None :
    
    for file in tqdm(os.listdir(data_path), desc=f"converting {data_type} data"):
        if file.endswith('.inkml'):
            filename = file.split('.')[0]
            inkml = file
            text_file = filename + "_coor.txt"
            # Parse inkml file
            inkml_path = os.path.join(data_path,inkml)
            tree = ET.parse(inkml_path)
            root = tree.getroot()
            # extract coordinate
            data = []
            for trace in root.findall("trace"):
                text = trace.text
                # get x,y coordinates and add 0 (pen down as default)
                coor = [p.split()[:2] for p in text.split(',')]
                coor = [[x,y,"0"] for x,y in coor]
                data.append(coor)
                # Add pen up indicator
                for trace in data[1:]:
                    trace[0][2] = "1"
                # create text file with coordinate & pen indicator
                text_path = data_path + text_file
                with open(text_path,"w") as t:
                    for trace in data:
                        for p in trace:
                            t.write(" ".join(p)+'\n')

def main(train_data_path, val_data_path):
    
    convert_inkml_to_points(train_data_path, data_type="train")
    convert_inkml_to_points(val_data_path, data_type="val")

if __name__=="__main__":
    
    train_data_path = "data/Training/"
    val_data_path = "data/Validation/" 

    main(train_data_path, val_data_path)