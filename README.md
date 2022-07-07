This is the code repository for [SereTOD Challenge](http://seretod.org/Challenge.html), co-located with [EMNLP2022 SereTOD Workshop](http://seretod.org)!

# Introduction
Task-oriented dialogue (TOD) systems are designed to assist users to accomplish their goals, and have gained more and more attention recently in both academia and
industry with the current advances in neural approaches. A TOD system typically consists of several modules, which track user goals to update dialog states, query a
task-related knowledge base (KB) using the dialog states, decide actions and generate responses. 

The purpose of this challenge is to invite researchers from both academia and industry to share their perspectives on building **se**mi-supervised and **re**inforced **TOD** systems and to advance the field in joint effort.
Hence, we refer to this challenge as the sereTOD challenge.
A shared task is organized for benchmarking and stimulating relevant researches. For the first sereTOD challenge, a large-scale TOD dataset is newly released, consisting of 100,000 real-world dialogs, where only 10,000 dialogs are annotated.

The task consists of two tracks:

* Information extraction from dialog transcripts (Track 1)   
* Semi-supervised task oriented dialog systems (Track 2)

**Organizers:** Zhijian Ou, Junlan Feng, Juanzi Li, Yakun Li, Hong Liu, Hao Peng, Yi Huang, Jiangjiang Zhao

# Important Dates (AOE)
| Date  | Item  |
| ---: | :--- |
| ~~July 1, 2022~~ | ~~Registration deadline for challenge~~ |
| ~~July 1, 2022~~ | ~~Training data release~~ |
| July 15, 2022 | ACL Rolling Review paper due (submit via [ARR](https://aclrollingreview.org/)) |
| August 14, 2022 | Evaluation data release |
| August 21, 2022 | Entry submission deadline |
| August 31, 2022 | Evaluation results announced |
| September 7, 2022 | Regular paper due (submit via [softconf](https://softconf.com/emnlp2022/seretod/)) |
| October 9, 2022 | Notification of paper acceptance |
| October 16, 2022 | Camera-ready papers due |
| December 7, 2022 | EMNLP 2022 Workshop Date |

# Important Links
* [SereTOD Challenge Website](http://seretod.org/Challenge.html)  
* [Challenge Description](http://seretod.org/SereTOD_Challenge_Description_v2.0.pdf)    
* [Track 1 Details](Track1/) (including baseline and evaluation code)
* [Track 2 Details](Track2/) (including baseline and evaluation code) 

If you publish experimental results with the MobileCS dataset or use the baseline models, please cite [this article](http://arxiv.org/abs/2207.02657):
```
@misc{ou2022achallenge,
      title={A Challenge on Semi-Supervised and Reinforced Task-Oriented Dialog Systems}, 
      author={
        Zhijian Ou and Junlan Feng and Juanzi Li and Yakun Li and Hong Liu and Hao Peng and Yi Huang and Jiangjiang Zhao},
      year={2022},
      eprint={2207.02657},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Rules
* The challenge website is http://seretod.org/Challenge.html . Teams should submit the registration form to seretod2022 (at) gmail (dot) com, which will be reviewed by the organizers. 
* Teams are required to sign an Agreement for Challenge Participation and Data Usage. Data will be provided to approved teams.
* For teams that participate in Track 1, the scores will be ranked according to the performance for Track 1. The teams can choose to participate only in Track 1.
* For teams that participate in Track 2, they can use the baseline system provided by the organizers or use the system developed by themselves for Track 1. The ranking is based on the performance for Track 2.
* Participants are allowed to use any external datasets, resources or pre-trained models which are publicly available.
* Participants are NOT allowed to do any manual examination or modification of the test data.

# Contact
For any questions, please feel free to contact: seretod2022 (at) gmail (dot) com

You are welcome to scan the following QR code to join the WeChat group (in Chinese), where we will distribute messages relevant to the Challenge.
<p align="center">
<img src="figs/ewm.png" alt="WeChat QR code" width="200" align="center"/>
</p>
