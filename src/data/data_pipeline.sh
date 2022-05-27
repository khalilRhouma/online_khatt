wget http://www.ccse.kfupm.edu.sa/dbs/onlinekhatt/Data/OnlineKHATTData.zip
unzip OnlineKHATTData.zip && rm -r OnlineKHATTData.zip SegmentedCharacters.zip
unrar x Training.rar data/ && rm -r Training.rar
unrar x Validation.rar data/ && rm -r Validation.rar

# creat points from inkml
python src/data/data_preparation.py 

# create features data
python src/data/data_preprocessing.py 

# move preprocessed data under data/ dir
cp -r preprocessed_data/* data/
rm -r preprocessed_data data/Training data/Validation