# Ambiguity-aware Emotion Recognition Leveraging Large Language Models
This repository contains the code and experimental setup for our research on emotion recognition using large language models (LLMs). Our study focuses on three databases (IEMOCAP, MSP-Podcast, and GoEmotion). We used zero-shot, few-shot technqiues to address the ambiguity problem in emotion recognition. 

## Experimental setups on datasets 
### IEMOCAP dataset
- Gemini.ipynb: basic zero-shot prompting
- Gemini_fewshot.ipynb: few-shot prompting with speech features
- Gemini_AudioFea.ipynb: zero-shot prompting with speech features
- Gemini_fs_noaudio.ipynb: few-shot prompting without speech features
- Gemini0context.ipynb: zero-shot prompting without context

### MSP-Podcast dataset
- msp.ipynb: zero-shot prompting with speech features
- msp_zs_cont10.ipynb: zero-shot prompting with context
- msp_withcontext.ipynb: zero-shot prompting with context
- msp_fs_audio.ipynb: few-shot prompting with speech features
- msp_fs_withcon.ipynb: few-shot prompting with context without speech features

### GoEmotion dataset
- goemotions_fs_clean.ipynb: few-shot prompting with 0 context 


## Evaluation 
- eval_metrics.py: functions of evaluation metrics methods
- Evaluation.ipynb: analysis on experiments performance
- IEMOCAP_script_impro.ipynb: comparison on the performance of script/impro datasets 


## LLM variance analysis
- msp_LLM_variation_analysis.ipynb: variation analyasis based on five identical runs




