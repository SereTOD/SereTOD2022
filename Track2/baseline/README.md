# Baseline for SereTOD Track2
This repository contains the baseline code for SereTOD Track2.
## Requirement
After you create an environment with `python 3.6`, the following commands are recommended to install the required packages
* pip install torch==1.5
* pip install transformers==3.5
* pip install nltk
* pip install tensorboard
* pip install tqdm
## Data Preprocessing
First you need to put raw data in `data` directory and rename it to `Raw_data.json`, then run
```
python preprocess.py
```
This script basically includes the following steps: 
1. Reconstruct the data so that users speak before the customer service in every turn;
2. Normalize the data;
3. Extract the user goal and local KB for every dialogue
## Training
The labeled data is splited into training set, validation set and test set with 8:1:1 ratio. You can train the dialog system with all labeled data
```
bash train.sh $DEVICE
```
`$DEVICE` can be "cpu" or GPU such as "cuda:0". 
## Testing
Only local KB and dialogue log are used in test set. You can perform end-to-end evaluation on the test set
```
bash test.sh $DEVICE $MODEL_PATH
```

