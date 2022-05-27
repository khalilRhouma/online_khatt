# online_khatt

Train and feature preparation are taken from [DeepOnKHATT](https://github.com/khalilRhouma/DeepOnKHATT) project
goal is to improve/optimize the code and fine-tune the model on other dataset

## Environment Setup
```
conda env create -f environment.yaml
```

## Data Preparation 
All handwriting samples have to be in the following format 
```
679.785826771654 70.0346456692913 0
679.785826771654 70.0346456692913 0
679.181102362205 68.4850393700787 0
678.727559055118 67.8047244094488 0
678.047244094488 67.7291338582677 0
................. ............... .
................. ............... .
................. ............... .
................. ............... .
676.573228346457 70.2236220472441 1
```
The first column is x coordinate, the second column is y coordinate and the third column is pen up indicator. \

All samples files should have label file (examples and jupyter notebook provided in features directory)

## Configuration

General configuration can be found in neural_network.ini file

## Training
To start training from scratch run the following command: \
```
python src/models/train.py --config neural_network.ini
```
make sure that neural_network.ini is under `src/configs/` directory

### Fine-tuning
To load the pre-trained model and continue training run the following command: \

```
python src/models/train.py --config neural_network.ini --path models/model.ckpt-i
```
