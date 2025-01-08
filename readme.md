# Ambiguity-aware Emotion Recognition Leveraging Large Language Models
This repository contains the code and experimental setup for our research on emotion recognition using large language models (LLMs). Our study focuses on three databases (IEMOCAP, MSP-Podcast, and GoEmotion). We used zero-shot, few-shot technqiues to address the ambiguity problem in emotion recognition. 

## File structure
- data_processing
    - load_data.py: load the original data and prepare then for experiments.
    - preprocessing.py: organize text and audio data.
    - prediction_processing.py: process the predictions from the LLM.
- Evaluation
    - eval_metrics.py: calculation methods of evaluation metrics including JS, BC, R_sqaure, ACC and ECE. 
    - eval_framework.py: evaluate the performance of the LLM.
- goemotions_experiment_sample.ipynb: 
- IEMOCAP_experiment_sample.ipynb
- msp_experiment_sample.ipynb
- prompt.py: Two types of shot prompt for LLM, including context and audio features. 

## Data
- IEMOCAP: https://sail.usc.edu/iemocap/
- MSP-Podcast: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html
- GoEmotion: https://arxiv.org/abs/2005.00547


