# Video Game Dialogue Summarization

Video game dialogue dataset: 
[VideoGameDialogueCorpusPublic](https://github.com/seannyD/VideoGameDialogueCorpusPublic)

Models can be found on:
[Huggingface](https://huggingface.co/models?pipeline_tag=summarization)

All scripts should be run with a modern version of Python 3 (3.10 or higher is preferred). \

Recommended to create a separate python environment
```
$ python3 -m venv env
$ source env/bin/activate
```
Some scripts require extra dependencies. Make sure to install all required ones from the requirements.txt. 
```
$ pip3 install -r requirements.txt
```
# USAGE AND OPTIONS
Includes to programs:
- dialogue_summarizer.py: summarize text
- calculate_rouge.py: calculate ROUGE metrics on generated summary with a reference summary

Contains two folders:
- `data`: data folder contains reference summaries expected to be in the following format:
  - data/{which_game}_summary_act{which_act}.txt','r',encoding='utf-8'
    - which_game: a game summarize i.e. `ff7`
    - which_act: which act in the game summary
- `pred`: pred folder contains generated summaries by the model, output format:
  - `pred/{which_game}act{which_act}_summary_pred_process({which_preprocess})_model({which_model})_length({max_dialogue_length}).txt'
    - which_game: for which game a summary has been made i.e. `ff7`
    - which_act: which act in the game summary
    - which_preprocess: which preprocess was used during the summary
    - which_model: which model was used for the summary
    - max_dialogue_length: the maximum dialogue length used to determine the summary.

Example to run commands
```
$ python3 dialogue_summarizer.py - m 0 -mdl 2048 - wa 1 -pp none
$ python3 calculate_rouge.py - m 0 -mdl 2048 - wa 1 -pp none
```

## General Options/Parameters for the programs
```
-m, -model                  select which model to use: 0:led-large-book-summary,
                            1:T5-Finetuned-Summarization-DialogueDataset
-mdl, -max_dialogue_length  max character length of dialogue summary, choices =[2048,4096]
-wg, -which_game            which game to summarize, choices = [1,2,3,4]
-wa, -which_act             which act to summarize, set 0 if summary over
                            entire story, choices = [0,1,2,3,4]
-pp, -preprocess            which preprocess to use.none: no preprocessing,
                            no_name: basic preprocessing, but remove names,
                            name_explicit: basic preprocessing, add name explicitly ass dialogue.
```  


