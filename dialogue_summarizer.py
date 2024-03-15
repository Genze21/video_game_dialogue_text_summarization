import torch
from transformers import pipeline

# select a model to use
which_model = 0
if which_model == 0:
  model_name = 'pszemraj/led-large-book-summary'
elif which_model == 1:
  model_name = 'Falconsai/text_summarization'
elif which_model == 2:
  model_name = 'chanifrusydi/t5-dialogue-summarization'
elif which_model == 3:
  model_name = 'pszemraj/long-t5-tglobal-xl-16384-book-summary'
elif which_model == 4:
  model_name = 'pszemraj/long-t5-tglobal-base-16384-book-summary'
elif which_model == 5:
  model_name = 'gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset'
elif which_model == 6:
  model_name = 'facebook/bart-large-cnn'

# retrieve the summarizer
summarizer = pipeline(
  "summarization",
  model_name,
  device=0 if torch.cuda.is_available() else -1,
)

# text
dialogue = """
"The opening sequence: A whirl of stars, humming softly, the view slowly drifting downwards and changing to show Aeris's face lit by Mako. She stands and walks out into the streets of Midgar as the camera pans back to show the city in full before again diving in, this time following a train as it pulls into a station in Sector 8. Biggs and Jessie leap off first, taking out the two guards stationed there. Wedge follows, and the three of them run ahead. Barret and Cloud leap down last.",
"C'mon newcomer. Follow me.", says Barret.
"Barret runs ahead. Cloud follows, taking out two guards who stand in his path. He joins Biggs, Wedge, and Jessie at the gate to the reactor complex, where Jessie is working to open the door.", says Barret.
"Wow! You used to be in SOLDIER all right! ... Not everyday ya find one in a group like AVALANCHE.", says Biggs.
"SOLDIER? Aren't they the enemy? What's he doing with us in AVALANCHE?", says Jessie.
"Hold it, Jessie. He WAS in SOLDIER. He quit them and now is one of us. Didn't catch your name...", says Biggs."
"""

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
print(result)

# print(summarizer(ARTICLE, max_length=1000, min_length=30, do_sample=False))