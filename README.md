# Read Between the Lines

![Project Poster](Submission_Combined_Malak_Miguel_Joao.jpg)

Authors: Malak Khan, Miguel Betrán Menz, João Fernandes Bárbara 

## Overview

This repository contains the implementation of three distinct architectures designed for a Multiple Choice Question Answering (MCQA) task. The goal is to select the correct option among four choices given a context and a question. The data is originaly from the ReClor dataset.

The project investigates whether simpler architectures or those trained from scratch with no external data can compete with large pre-trained models fine-tuned the small available dataset.

## Repository Structure

The project is organized by model architecture. All necessary data is included in the assignment_data directory.

```
├── assignment_data/          # Contains train.csv and test.csv
├── source_baseline/          # Code for Baseline Method: MALAK ADD DESCRIPTION
├── source_minilreasoner/     # Code for Method A: Transformer-based MiniLReasoner
├── source_rand_dagn/         # Code for Method B: Rand-DAGN (Graph Network)
├── source_bi_lstm/           # Code for Method C and C*: Bi-LSTM with Attention
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

## Installation

### Clone the repository:

```
git clone https://github.com/JoaoBarbara5/nlp-logic-model.git
cd nlp-logic-model
```

### Install dependencies: 

Ensure you have Python installed, then run:

```pip install -r requirements.txt```

## Running the Models

Each model is self-contained in its respective folder. No extra downloads are required. The submission files are generated in the model's folder by running each file.

## NOTE:

For the DAGN Architecture, ONLY the RandomEmbedding module/class is written by us. The rest of the model architecture is directly cited from:

https://github.com/Eleanor-H/DAGN

@InProceedings{zhang2021video,
author = {Huang, Yinya and Fang, Meng and Cao, Yu and Wang, Liwei and Liang, Xiaodan},
title = {DAGN: Discourse-Aware Graph Network for Logical Reasoning},
booktitle = {NAACL},
year = {2021}
} 
