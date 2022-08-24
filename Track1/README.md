# SereTOD Track1: Information Extraction from dialog transcripts
This repository contains the task, evaluation, data and baseline codes for SereTOD Track1. 
# Update 
**2022.08.24** Update `post_process.py`: fix offset computation bugs for labels. The lastest baseline results are: **F1 (entity) 33.45, F1 (Triple) 34.94**. \  
**2022.08.24** Update `post_process.py`: fix `数据业务` replace bugs. The lastest baseline results are: **F1 (entity) 32.85, F1 (Triple) 34.94**. \   
**2022.08.22** Update evaluation script: 1. Use `offset` for evaluation of entities. 2. Use `mention` for evaluation of triples. 3. Add `turn_id` for more precise evaluation. 4. Delete golden labels per step to avoid influence of duplicate predictions. The lastest baseline results are: **F1 (entity) 32.86, F1 (Triple) 34.75**. \
**2022.08.19** Update evaluation script: 1. Use `mention` instead of `offset` for evaluation 2. Add `post_process.py` to filter whitespace and `_` before evaluation. The lastest baseline results are: **F1 (entity) 38.16, F1 (Triple) 38.40**. 

# Task    
In a task-oriented dialog system, after dialog state tracking, the system needs to query a task-related knowledge base. Given a mix of labeled and unlabeled dialog transcripts, Track 1 examines the task of training information extraction models to construct the “local” knowledge base for each dialog, which will be needed in training TOD systems in Track 2. Therefore, we define two sub-tasks:  
1) Entity extraction. This sub-task is to extract entities with their corresponding concepts. In real-life dialogs, an entity
may be mentioned in different surface forms, which need to be extracted. For example, “50元流量包” may have a number of different mentions in a multi-turn dialog: “50元那个业务” , “那个流量包” . Thus, entity extraction for the MCSD dataset is more challenging than classic NER tasks, due to the informal, verbalized and loose form of the customer-service dialogs.  
2) Slot filling. This sub-task is to extract slot values for entity slots (i.e., attributes). It is formulated as a sequence labeling task for the pre-defined slots in the schema. For example, in sentence “10GB套餐业务每月的费用是50块钱。” , “每月的费用是50块钱” will be labeled as plan price slot. An entity may have several mentions in a dialog, and the slots and values for an entity may scatter in multi-turn dialogs. Thus, the task requires entity resolution and assigning slot-value pairs to the corresponding entity. After entity extraction and slot filling, a local knowledge base (KB) will be constructed with all extracted entities with their attributes for each dialog.   
# Evaluation  
Given a dialog in testing, the trained information extraction model is used to extract entities together with slot values. We will evaluate and rank the submitted models by the extraction performance on test set. The evaluation metrics are Precision, Recall and F1.  
1) For entity extraction, the F1 is calculated at entity mention level: an entity mention is extracted correctly if and only if the mention span of the entity is labeled as the corresponding entity-type (i.e., concept).   
2) For slot filling, the F1 is calculated at triple level: an entity-slot-value triple is extracted correctly if and only if 1) the mention span of the slot value is labeled as the corresponding slot type. 2) the slot-value pair is correctly assigned to the corresponding entity.  

For entity extraction, the participants need to submit all the predicted mentions with their types. For slot filling, the participants need to submit the extracted entities with entity resolution. Each extracted entity may contain multiple mentions and is represented as a set of entity-slot-value triples. The performance of slot filling is measured by finding the best match between the extracted entities and the golden labeled entities using the [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) and calculating the F1.    

The average F1 scores of entity extraction and slot filling will be the ranking basis on leaderboard. We will provide the following scripts and tools for the participants: 1) Baseline models for both sub-tasks; 2) Evaluation scripts to calculate the metrics.



# Baseline 
The folder provides a pipeline for extracting entities and their corresponding properties (slots) and values. In general, the pipeline contains 4 components.  
**1. Entity Extraction**  
We implement a sequence labeling model to extract entity mentions in dialog utterances. Using this 
component, we get entity mentions from utterances.  
**2. Entity Coreference**  
After entity extraction, we need to cluster the mentions of the same entity into a single cluster. Therefore, we provide the component to conduct entity coreference resolution.  
**3. Slot Filling**  
After entity extraction and entity coreference resolution, we extract properties (slots) using a
sequence labeling method.  
**4. Entity Slot Alignment**  
After extracting entities and slots, we need to align them (i.e., assign slots to the coresponding entities). 
The entire procedure of the pipeline is as follows:
```
Entity Extractioin --> Entity Coreference --> Slot Filling --> Entity Slot Alignment
```
The code for each component is placed in the corresponding folder. 


### Setup
First, install all the requirements:
```Shell
pip install -r requirements.txt 
```
Then, run all the components in order to get the prediction:
```Shell
bash run_all.sh
```
Finally, use the following script to get the final submissions:
```Shell
python get_submissions.py
```
You should place the `data` folder in the `baseline/.`.

### Submission Format
After runing the pipeline, you can use `get_submissions.py` to get the final submission file.
The submission file should be a json formatted file, and the format is as follows:
```Json
[
    { // a doc
        "id": "2aa131d5143bddb3772f595292987780", // doc id,
        "entities": [ // extracted entities 
            {
                "id": "ent-2",
                "name": "一百三十八的那个套餐",
                "type": "主套餐",
                "position": [
                    10,
                    20
                ],
                "utterance_id": 3 // index of the utterance 
            }
        ]
        "triples": [
            {
                "ent-name": "None", // ent name is not needed, you can set it "None"
                "ent-id": "ent-2",
                "value": "一百三十八",
                "prop": "业务费用",
                "position": [
                    10,
                    15
                ],
                "utterance_id": 3
            }
        ]
    }
]
```

### Evaluation and Results
We use micro-F1 as the basic metric. Before evaluation, we run Hungarian Algorithm on the submissions to 
get best assignments between predicted entities and golden entities. The evaluation script `eval_script.py` is also provided in the repo.  
We random sample 1,000 instances as the test set and evaluate our baseline on the test set. The results are:
**F1 (entity) 38.16, F1 (Triple) 38.40**. The results are relatively low, which indicates the task is challenging and needs more powerful models. 
