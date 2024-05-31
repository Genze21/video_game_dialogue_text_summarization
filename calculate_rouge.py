# https://huggingface.co/spaces/evaluate-metric/rouge

import evaluate # Load the ROUGE metric
import argparse

def main(args):
  which_model = args.m            # which model to use
  max_dialogue_length = args.mdl  # max character length of dialogue summary
  which_preprocess = args.pp      # which preprocess to use
  which_game = args.wg            # which game to preprocess
  which_act = args.wa             # which act to summarize in ff7

  rouge = evaluate.load('rouge')

  # create a list with generated summary from model
  predictions = []
  with open(f'pred/{which_game}act{which_act}_summary_pred_process({which_preprocess})_model({which_model})_length({max_dialogue_length}).txt', 'r', encoding='utf-8') as prediction_text:
    tmp = ''
    for line in prediction_text:
      tmp += line
    predictions.append(line)

  # create a list of list with reference summaries
  references = []
  lines = []
  with open(f'data/{which_game}_summary_act{which_act}.txt','r',encoding='utf-8') as reference_text:
    for line in reference_text:
      lines.append(line)
    references.append(lines)
    results = rouge.compute(predictions=predictions, references=references)
    print(f"rouge1: {results['rouge1']:.3f} \t rouge2: {results['rouge2']:.3f} \t rougeL: {results['rougeL']:.3f} \t rougeLsum: {results['rougeLsum']:.3f}")

# read arguments from command line
def read_args():
  parser = argparse.ArgumentParser(description='Hedgehog <3')
  
  parser.add_argument('-m', '-model', type = int, help='model used: 0:led-large-book-summary,1:T5-Finetuned-Summarization-DialogueDataset', choices=[0,1], default=1)
  parser.add_argument('-mdl', '-max_dialogue_length', type = int, help='max character length of dialogue summary', choices=[2048,4096], default=4096)
  parser.add_argument('-wg', '-which_game', help='which game to summarize', default='ff7')
  parser.add_argument('-wa', '-which_act', type = int, help='act to summarize in ff7', choices=[0,1,2,3,4], default=1)
  parser.add_argument('-pp', '-preprocess', help='which preprocess', choices=['none', 'no_name','name_explicit'], default='none')
  
  return parser.parse_args()

if __name__ == '__main__':
  args = read_args()
  main(args)