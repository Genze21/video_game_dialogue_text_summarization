import torch
from transformers import pipeline
from textsum.summarize import Summarizer
import json

def main(max_dialogue_length):
  # select a model to use
  which_model = 1
  if which_model == 0:
    model_name = 'pszemraj/led-large-book-summary' # slow
  elif which_model == 1:
    model_name = 'gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset' # fast

  dialogue_text = ''  # text to be summarized
  with open ("data/ff7_data.json") as dialogue_data:
    data = json.load(dialogue_data)

  # determine action on a given line of text
  # 0=action(dialogue not by a character)
  # 1=location
  # 2=choice
  # 3=dialogue(dialogue by a character)
  def which_action(line):
    if 'ACTION' in line:
      return 0
    elif 'LOCATION' in line:
      return 1
    elif 'CHOICE' in line:
      return 2
    else:
      return 3

  # characters that can/should be removed from string
  remove_characters = [('{', ''),('}', '')]

  # create empty file
  with open(f'data/ff7_summary_pred_{which_model}.txt', 'w',encoding="utf-8") as pred_file:
    pass

  # initialize textsum method to summarize
  summarizer = Summarizer(
      model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
      token_batch_length=max_dialogue_length,  # tokens to batch summarize at a time, up to 16384,
  )

  for counter,line in enumerate(data['text']):
    if counter < 200:
      # convert dict to str
      line = str(line)

      # filter on certain action
      action =  which_action(line)

      # TODO fix CHOICE(2) action
      if(action in [0,1,3]):

        # remove characters for formatting
        for char, replacement in remove_characters:
          if char in line:
            line = line.replace(char, replacement)

        # split line into action 
        action, sentence = line.split(":",1)      

        if (len(dialogue_text) <= max_dialogue_length):
          dialogue_text += sentence
        else:
          print(dialogue_text)
          out_str = summarizer.summarize_string(dialogue_text)
          with open(f'data/ff7_summary_pred_{which_model}.txt', 'a',encoding="utf-8") as pred_file:
            pred_file.write(f"{out_str} \n")
            
          dialogue_text = ''

  # summarize last bit of dialogue
  if len(dialogue_text) != 0:
    out_str = summarizer.summarize_string(dialogue_text)
    with open(f'data/ff7_summary_pred_{which_model}.txt', 'a',encoding="utf-8") as pred_file:
      pred_file.write(f"{out_str} \n")

if __name__ == '__main__':
  max_dialogue_length = 4096  # max character length of dialogue summary
  main(max_dialogue_length)