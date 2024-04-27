import torch
from transformers import pipeline
from textsum.summarize import Summarizer
import time


# select a model to use
which_model = 0
if which_model == 0:
  model_name = 'pszemraj/led-large-book-summary' # slow
elif which_model == 1:
  model_name = 'Falconsai/text_summarization' # fast
elif which_model == 2:
  model_name = 'chanifrusydi/t5-dialogue-summarization' # fast
elif which_model == 3:
  model_name = 'pszemraj/long-t5-tglobal-xl-16384-book-summary' 
elif which_model == 4:
  model_name = 'pszemraj/long-t5-tglobal-base-16384-book-summary' # slow
elif which_model == 5:
  model_name = 'gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset' # fast
elif which_model == 6: # not working
  model_name = 'facebook/bart-large-cnn'

# retrieve the summarizer
summarizer = pipeline(
  "summarization",
  model_name,
  device=0 if torch.cuda.is_available() else -1,
)

start_time = time.time()
# read the input dialogue
dialogue = """ """
with open('data/input_text_example_ff7.txt', 'r',encoding='utf-8') as dialogue_text:
  for line in dialogue_text:
    dialogue += line

dialogue_time = time.time()

# create the results and set the parameters
result = summarizer(
    dialogue,
    min_length=16,
    max_length=256,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    early_stopping=True,
)
# show result
sum_time_1 = time.time()

# textsum method to summarize
summarizer = Summarizer(
    model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
    token_batch_length=4096,  # tokens to batch summarize at a time, up to 16384
)

sum_time_2 = time.time()
out_str = summarizer.summarize_string(dialogue)

with open(f'data/ff7_summary_pred_{which_model}.txt', 'w',encoding="utf-8") as pred_file:
  pred_file.write(out_str)



print(result)
print('--------------')
print(f"summary: {out_str}")

print('--------------')
print(f'time for reading dialogue: \t {dialogue_time-start_time}')
print(f'time for sum1: \t {sum_time_1-dialogue_time}')
print(f'time for sum2: \t {sum_time_2-sum_time_1}')