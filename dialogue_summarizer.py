import torch
from transformers import pipeline
from textsum.summarize import Summarizer
import json
import argparse

# determine action on a given line of text
# 0=action(dialogue not by a character)
# 1=location
# 2=choice
# 3=dialogue(dialogue by a character)
def which_action(line):
  if 'CHOICE' in line:
    return 2
  elif 'ACTION' in line:
    return 0
  elif 'LOCATION' in line:
    return 1
  else:
    return 3

# removes some character from a string for better formatting
def remove_characters(line, separator_1='{',separator_2='}'):
  # characters that can/should be removed from string
  characters_to_remove = [(separator_1, ''),(separator_2, '')]

  # remove characters for better formatting
  for char, replacement in characters_to_remove:
    if char in line:
      line = line.replace(char, replacement)

  return line

# applies a preprocess to the selected dialogue
# none: no preprocessing
# no_name: remove names
# name_explicit: add names explicitly i.e. "xxx", says char A
def apply_preprocess(which_preprocess,sentence,dialogue_type,action):
  dialogue_text = ''
  # add dialogue to be summarized
  if which_preprocess == "no_name" or which_preprocess == "none" or (which_preprocess == "name_explicit" and dialogue_type != 3):
    dialogue_text += sentence
  # add names explicitly 
  elif which_preprocess == "name_explicit" and dialogue_type == 3:
    dialogue_text += f"{sentence}, says {action}."
  
  return dialogue_text

def main(args):
  which_model = args.m            # which model to use
  max_dialogue_length = args.mdl  # max character length of dialogue summary
  which_preprocess = args.pp      # which preprocess to use
  which_act = args.wa             # which act to summarize in ff7

  # select a model to use
  if which_model == 0:
    model_name = 'pszemraj/led-large-book-summary' # slow
  elif which_model == 1:
    model_name = 'gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset' # fast

  dialogue_text = ''  # text to be summarized
  with open (f"data/ff7_data_act{which_act}.json") as dialogue_data:
    data = json.load(dialogue_data)

  # create empty file
  with open(f'pred/ff7act{which_act}_summary_pred_process({which_preprocess})_model({which_model}).txt', 'w',encoding="utf-8") as pred_file:
    pass

  # initialize textsum method to summarize
  summarizer = Summarizer(
    model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
    token_batch_length=max_dialogue_length,  # tokens to batch summarize at a time, up to 16384,
  )

  for counter,line in enumerate(data['text']):
    # if counter < 200:
      # convert dict to str
      line = str(line)

      # filter on certain action
      dialogue_type =  which_action(line)

      if(dialogue_type in [0,1,2,3]):
        if which_preprocess == "none":
          sentence = line
          action = 'none'
        else:
          line = remove_characters(line)
          # split line into action 
          action, sentence = line.split(":",1)    

        # CHOICE(2) action, selects the first choice available in splits it into separate dialogue prompts. 
        if dialogue_type == 2:
          line_split = sentence.split('],')       # split choices
          line_split = remove_characters(line_split[0],'[',']') 
          choices_split = line_split.split("',")  # split prompts
          if len(choices_split)==1:
            choices_split = line_split.split('",')

        # add line of dialogue until limit
        if (len(dialogue_text) <= max_dialogue_length):
          if dialogue_type != 2:
            dialogue_text += apply_preprocess(which_preprocess,sentence,dialogue_type,action)
          if dialogue_type == 2:
            for choice in choices_split:
              dialogue_text += apply_preprocess(which_preprocess,choice,dialogue_type,action)

        # sent dialogue to summarizer and reset the dialogue
        else:
          # TODO add context to the dialogue
          # if which_preprocess == "context":
          #   pass
          print(dialogue_text)
          out_str = summarizer.summarize_string(dialogue_text)
          with open(f'pred/ff7act{which_act}_summary_pred_process({which_preprocess})_model({which_model}).txt', 'a',encoding="utf-8") as pred_file:
            pred_file.write(f"{out_str} \n")
            
          dialogue_text = ''

  # summarize last bit of dialogue
  if len(dialogue_text) != 0:
    out_str = summarizer.summarize_string(dialogue_text)
    with open(f'pred/ff7act{which_act}_summary_pred_process({which_preprocess})_model({which_model}).txt', 'a',encoding="utf-8") as pred_file:
      pred_file.write(f"{out_str} \n")

# read arguments from command line
def read_args():
  parser = argparse.ArgumentParser(description='Hedgehog <3')
  
  parser.add_argument('-m', '-model', type = int, help='model used: 0:led-large-book-summary,1:T5-Finetuned-Summarization-DialogueDataset', choices=[0,1], default=1)
  parser.add_argument('-mdl', '-max_dialogue_length', type = int, help='max character length of dialogue summary', choices=[2048,4096], default=4096)
  parser.add_argument('-wa', '-which_act', type = int, help='act to summarize in ff7', choices=[1,2,3,4], default=1)
  parser.add_argument('-pp', '-preprocess', help='which preprocess', choices=['none', 'no_name','name_explicit', 'context'], default='none')
  
  return parser.parse_args()

if __name__ == '__main__':
  args = read_args()
  main(args)