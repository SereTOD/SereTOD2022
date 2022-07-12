# SereTOD Track2: Task-Oriented Dialog Systems
This repository contains the evaluation and baseline code for SereTOD Track2.

Most existing TOD systems require not only large amounts of annotations of dialog states and dialog acts (if used), but also a global knowledge base (KB) covering all public knowledge and all personal information in the domain, which are both difficult to obtain at the research stage. Compared with previous work, the task in Track2 has two main characteristics:
* There's no global KB but only a local KB for each dialog, representing the unique information for each user, e.g., the user's package plan and remaining phone charges.
*  Only a proportion of the dialogs is annotated with intents and local KBs. The teams are encouraged to utilize a mix of labeled and unlabeled dialogs to build a TOD system.

# Task Definition
The basic task for the TOD system is, for each dialog turn, given the dialog history, the user utterance and the local KB, to predict the user intent, query the local KB and generate appropriate system intent and response according to the queried information. 
For every labeled dialog, the annotations consist of user intents, system intents and a local KB. The local KB is obtained by collecting the entities and triples annotated for Track 1.
For unlabeled dialogs, there are no such annotations.

# Evaluation
In order to measure the performance of TOD systems, both automatic evaluation and human evaluation will be conducted. 
For automatic evaluation, metrics include Precision/Recall/F1 score, Success rate and BLEU score.  P/R/F1 are calculated for both predicted user intents and system intents.
Success rate is the percentage of generated dialogs that achieve user goals. BLEU score evaluates the fluency of generated responses.

We will perform human evaluation for different TOD systems, where real users interact with those systems according to randomly given goals. 
For each dialog, the user will score the system on a 3-point scale (0, 1, or 2) by the following 3 metrics. Three scales (0, 1 and 2) denote three degrees - not at all, partially and completely, respectively.
* Success. This metric measures if the system successfully completes the user goal by interacting with the user;
* Coherency. This metric measures whether the system's response is logically coherent with the dialogue context;
* Fluency. The metric measures the fluency of the system's response.

The average scores from automatic evaluation and human evaluation will be the main ranking basis on leaderboard.
We will provide the following scripts and tools for the participants: 1) A baseline system; 2) Evaluation scripts to calculate the corpus-based metrics.

## Submission Format
The examples of hidden test data are shown in [test_example.json](./baseline/Track2_data/test_example.json).
Participants will organize their results into the same format as [result_example.json](./baseline/Track2_data/result_example.json), where they need to use the generated user intent (用户意图-生成), system intent (客服意图-生成) and system response (客服-生成) to fill in the corresponding blank keys.

# Data and Baseline
In this challenge task, participants will use [SereTOD dataset](../data/) to build a TOD system. Note that there are only annotations of intents, entities and triples in the dataset, so it is neccessary to **extract the local KB and user goal** for each dialogue. The extraction script and baseline code can be seen in [baseline](./baseline/).
