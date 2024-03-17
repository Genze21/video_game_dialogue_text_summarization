# input: dialogue from data.json file
# output: dialogue with a given format

import json

with open ("data/ff7_data.json") as dialogue_data:
  data = json.load(dialogue_data)

def check_string(line):
  if 'LOCATION' in line:
    return False
  if 'ACTION' in line:
    return False
  else:
    return True

for counter,line in enumerate(data['text']):
  if counter < 10:
    if check_string(line):
      print(str(line))
      