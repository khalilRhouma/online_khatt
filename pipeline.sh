# Download and unzip data
wget http://www.ccse.kfupm.edu.sa/dbs/onlinekhatt/Data/OnlineKHATTData.zip
unzip OnlineKHATTData.zip && rm -r OnlineKHATTData.zip SegmentedCharacters.zip
unrar x Training.rar data/ && rm -r Training.rar
unrar x Validation.rar data/ && rm -r Validation.rar

# download trained chekpoints
gdown --id  1Z_gzzWVjskv_1JqErGuz8ZVfCSNaC3VY --output models/models.zip
unzip models/models.zip -d models/ && rm -r models/models.zip

## Data preparation and preprocessing
# creat points from inkml
python src/data/data_preparation.py
# create features data
python src/data/data_preprocessing.py
# move preprocessed data under "data" dir
cp -r preprocessed_data/* data/
rm -r preprocessed_data data/Training data/Validation

## Training
# Fine-tune the model from ckpt-14 and save logs
python src/models/train.py --config neural_network.ini --path models/model.ckpt-14 |& tee training.log
