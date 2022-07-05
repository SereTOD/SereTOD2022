# SereTOD Dataset: Mobile Customer-Service Dialog (MobileCS)

This directory contains the official dataset for [SereTOD Challenge](../README.md).
The evaluation dataset will be available later.

## Data Description
An important feature for our challenge is that we release around 100,000 dialogs (in Chinese), which come from real-world dialog transcripts between real users and
customer-service staff from China Mobile. We call this dataset as MCSD (mobile customer-service dialog) dataset. To the best of our knowledge, MCSD is not only the largest publicly available multi-domain TOD dataset, but also consists of real-life data (namely collected in real-world scenarios).  
We provide only 10,000 dialogs are labeled by crowdsourcing, while the remaining 90,000 dialogs are unlabeled. The teams are required to use this mix of labeled and unlabeled data to train information extraction models (Track 1), which could provide a knowledge base for Track 2, and train TOD systems (Track 2), which could work as customer-service bots. We put aside 5,000 dialogs as evaluation data.

## Data Formats
We provide some dialogue examples in [example.json](example.json). You can access the full data by contacting us by [email](SereTod2022@gmail.com). The data includes the list of instances each of which is a dialogue between users and the customer service. Each instance is a list of the following turn objects:
* Speaker ID: the speaker of the dialogue, such as "[SPEAKER 1]" and "[SPEAKER 2]"
* Intents: the intent of each speaker in this turn. "用户意图" represents the user intent, "客服意图" represents the system intent
* Information: including the entities and triples mentioned in this turn


    
