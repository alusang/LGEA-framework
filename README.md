# Publication

This work has been accepted and published:

### Breaking the Noise Barrier: LLM-Guided Semantic Filtering and Enhancement for Multi-Modal Entity Alignment

Chenglong Lu, Chenxiao Li, Jingwei Cheng, Yongquan Ji, Guoqing Chen, Fu Zhang

Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing

[PDF](https://aclanthology.org/2025.emnlp-main.1684.pdf) | [DOI](10.18653/v1/2025.emnlp-main.1684)  

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
  
## [Update] Data Processing

The dataset links above all provide access to the complete data. However, some datasets require additional preprocessing. EA researchers may also use partially preprocessed data provided by others. The required raw data can be found in the data loading files — simply preprocess the data into the expected format and load it accordingly. The overall preprocessing pipeline is listed below.

### 1. Visual Modality

The visual part consists of two components:

- **Image captions and masks**  
  First, generate image captions using `batch_blip_image_captioning.py`.  
  Then, use the generated captions together with an LLM in `gene_mask.py` to obtain the corresponding mask values.

- **Visual embeddings**  
  Extract embedding vectors from images.  

  You may either:
  - use existing preprocessed visual embeddings (e.g., commonly used ResNet features), or
  - re-extract embeddings using other vision models.

### 2. Structure, Attribute, and Relation Information

- **Relation and structural information**  
  The required format for these two modalities remains unchanged and can be directly loaded through the data loading files.

- **Attribute information**  
  For each entity, remove the entity name from its attribute triples and concatenate the remaining attribute-value pairs into the following format:

  ```python
  "eid": "[(att1, value1), (att2, value2), ...]"

Then, use `sum_embed.py` to perform summarization and embedding generation, producing the final attribute embedding vectors.

### 3. Model Input

Load the following processed data into the model:

- masks
- visual embeddings
- attribute embeddings
- relation information
- structural information

### Note

Some experimental code artifacts from earlier research stages (e.g., VAE-related modules) are still retained in the repository. These can be safely ignored and do not affect the correctness of the system or the reported results.

# Citation

If you find this code to be useful for your research, please consider citing.

```
@inproceedings{lu2025breaking,
  title={Breaking the Noise Barrier: LLM-Guided Semantic Filtering and Enhancement for Multi-Modal Entity Alignment},
  author={Lu, Chenglong and Li, Chenxiao and Cheng, Jingwei and Ji, Yongquan and Chen, Guoqing and Zhang, Fu},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  url={https://aclanthology.org/2025.emnlp-main.1684/},
  doi={10.18653/v1/2025.emnlp-main.1684},
  pages={33141--33155},
  year={2025}
}
```
