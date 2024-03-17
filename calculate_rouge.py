# https://huggingface.co/spaces/evaluate-metric/rouge

import evaluate # Load the ROUGE metric

rouge = evaluate.load('rouge')

# create a list with generated summary from model
predictions = []
with open('data/ff7_summary_pred_0.txt', 'r', encoding='utf-8') as prediction_text:
  for line in prediction_text:
    predictions.append(line)

# create a list of list with reference summaries
# TODO make reference text as one item
references = []
lines = []
with open('data/ff7_summary_act_1.txt','r',encoding='utf-8') as reference_text:
  # print(reference_text.read())
  for line in reference_text:
    lines.append(line)
  references.append(lines)
  results = rouge.compute(predictions=predictions, references=references)
  print(results)


