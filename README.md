# LGEA-Framework

This repository contains the implementation of **LGEA** for multi-modal entity alignment codes.

## Data Sources

- **DBP15K**
  - Text data: https://github.com/zjukg/MEAformer
  - Vision raw data: https://github.com/zjukg/MEAformer In Dataset section, a Baidu Cloud file
- **FB & DB & YG**: https://github.com/mniepert/mmkb

> Note: The part data needs to be preprocessed and loaded separately; the raw datasets are not included in this repository.

## Usage

1. **Data Processing**  
   - Preprocess the raw data according to the dataset instructions, then load it into the directories expected by the code.

2. **Running the Topic Model**  
   - Experiments related to the topic model can be run using the provided shell script
