# SereTOD Track1: Information Extraction from dialog transcripts
This repository contains the task, evaluation, data and baseline codes for SereTOD Track1. 
# Task    
In a task-oriented dialog system, after dialog state tracking, the system needs to query a task-related knowledge base. Given a mix of labeled and unlabeled dialog transcripts, Track 1 examines the task of training information extraction models to construct the “local” knowledge base for each dialog, which will be needed in training TOD systems in Track 2. Therefore, we define two sub-tasks.  
1) Entity extraction. This sub-task is to extract entities with their corresponding concepts, which are mentioned in a dialog session. In real-life dialogs, an entity
may be mentioned in different surface forms, which need to be extracted. For example, “50元流量包” may have a number of different mentions in a multi-turn dialog: “50元那个业务” , “那个流量包” . Thus, entity extraction for the MCSD dataset is more challenging than classic NER tasks, due to the informal, verbalized and loose form of the customer-service dialogs.  
2) Slot filling. This sub-task is to extract slot values for entity slots (i.e., attributes). It is formulated as a sequence labeling task for the pre-defined slots in
the schema. For example, in sentence “10GB套餐业务每月的费用是50块钱。” , “每月的费用是50块钱” will be labeled as plan price slot. An entity may have several mentions in a dialog, and the slots and values for an entity may scatter in multi-turn dialogs. Thus, the task requires entity resolution and assigning slot-value pairs to the corresponding entity.   
# Evaluation  
Given a dialog in testing, the trained information extraction model is used to extract entities together with slot values. We will evaluate and rank the submitted models by the extraction performance on test set. The evaluation metrics are Precision, Recall and F1.  
1) As for entity extraction, the metrics are at entity level: an entity is extracted correctly if and only if the mention span of the entity is labeled as the corresponding entity type (i.e., concept).  
2) As for slot filling, the metrics are at triple level: an entity-slot-value triple is extracted correctly if and only if 1) the mention span of the slot value is labeled as the corresponding slot type. 2) the slot-value pair is correctly assigned to the corresponding entity.  
# Data  
An important feature for this shared task is that we release around 100,000 dialogs (in Chinese), which come from real-world dialog transcripts between real users and
customer-service staff from China Mobile, with privacy information anonymized. We call this dataset as MCSD (mobile customer-service dialog) dataset, which differs
from existing TOD datasets in both size and nature significantly. To the best of our knowledge, MCSD isnot only the largest publicly available multi-domain TOD dataset, but also consists of real-life data (namely collected in real-world scenarios). 
We provide only 10,000 dialogs are labeled by crowdsourcing, while the remaining 90,000 dialogs are unlabeled. The teams are required to use this mix of labeled and unlabeled data to train information extraction models (Track 1), which could provide a knowledge base for Track 2, and train TOD systems (Track 2), which could work as customer-service bots. We put aside 5,000 dialogs as evaluation data.
