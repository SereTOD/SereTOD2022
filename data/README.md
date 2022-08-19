# Mobile Customer-Service Dialog Dataset (MobileCS)

This directory contains the official dataset for [SereTOD Challenge](../README.md).
The evaluation data will be released later.

## Data Description
An important feature for SereTOD challenge is that we release around 100,000 dialogs (in Chinese), which come from real-world dialog transcripts between real users and
customer-service staffs from China Mobile, with privacy information anonymized. 
We call this dataset as **MobileCS** (mobile customer-service) dialog dataset, which differs from existing TOD datasets in both **size** and **nature** significantly.
To the best of our knowledge, MobileCS is not only the largest publicly available TOD dataset, but also consists of real-life data (namely collected in real-world scenarios). For comparison, the widely used MultiWOZ dataset consists of 10K dialogs and is in fact simulated data (namely collected in a Wizard-of-Oz simulated game).

A schema is provided, based on which 10K dialogs are labeled by crowdsourcing. The remaining 90K dialogs are unlabeled.
The teams are required to use this mix of labeled and unlabeled data to train information extraction models (Track 1), which could provide a knowledge base for Track 2, and train TOD systems (Track 2), which could work as customer-service bots.
We put aside 1K dialogs as evaluation data. More details can be found in [Challenge Description](http://seretod.org/SereTOD_Challenge_Description_v2.0.pdf) .

## Data Format
We provide some dialog examples in [example.json](example.json), consisting of 10 dialogs.
The entire MobileCS dataset is provided to registered teams.

The dataset contains a list of instances, each of which is a dialog between a user and a customer-service staff. Each dialog is a list of turns. Each turn includes the following objects:
* Speaker ID: the speaker of the dialogue, such as "[SPEAKER 1]" and "[SPEAKER 2]"
* Intents: the intent of each speaker in this turn. "用户意图" represents the user intent, "客服意图" represents the system intent
* Information: including the entities and triples mentioned in this turn