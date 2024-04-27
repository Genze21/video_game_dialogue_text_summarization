# input: dialogue from data.json file
# output: dialogue with a given format

import json
def text_summary(max_dialogue_length):
  dialogue_text = ''  # text to be summarized
  with open ("data/ff7_data.json") as dialogue_data:
    data = json.load(dialogue_data)

  # determine action on a given line of text
  # 0=location,1=action,2=choice,3=dialogue
  def which_action(line):
    if 'LOCATION' in line:
      return 0
    elif 'ACTION' in line:
      return 1
    elif 'CHOICE' in line:
      return 2
    else:
      return 3

  # characters that can/should be removed from string
  remove_characters = [('{', ''),('}', '')]

  for counter,line in enumerate(data['text']):
    if counter < 200:
      # convert dict to str
      line = str(line)

      # filter on certain action
      action =  which_action(line)

      # TODO fix CHOICE(2) action
      if(action in [0,1]):
        # print(action)

        # remove characters for formatting
        for char, replacement in remove_characters:
          if char in line:
            line = line.replace(char, replacement)

        # split line into action 
        action, sentence = line.split(":",1)      

        if (len(dialogue_text) <= max_dialogue_length):
          dialogue_text += sentence
          print(len(dialogue_text))
        else:
          # pass dialogue to summary and reset 
          print(dialogue_text)
          dialogue_text = ''
          print('limit reached')
          continue

        # print(str(line))
        # print(f'action: {action} \t sentence:{sentence}')
      

if __name__ == '__main__':
  max_dialogue_length = 4096  # max character length of dialogue summary
  text_summary(max_dialogue_length)