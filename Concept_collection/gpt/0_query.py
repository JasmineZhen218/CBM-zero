import os
import openai
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser(description="GPT")
parser.add_argument("--data_name", help="data_name", default = 'food101')
parser.add_argument("--concept_source", help="data_name", default = 'food101_labo')
parser.add_argument("--gpt_model", help="gpt_model", default = 'gpt-4o')
args = parser.parse_args()

data_name = args.data_name
concept_source = args.concept_source
gpt_model = args.gpt_model

with open("asset/concept_bank/{}.txt".format(concept_source)) as f:
    concepts = f.readlines()
f.close()
with open("asset/class_names/{}.txt".format(data_name), "r") as f:
    classes = f.readlines()
    classes = [c.strip() for c in classes]
f.close()
print("{} concepts and {} classes".format(len(concepts), len(classes)))


if len(concepts)>50:
    Concepts_str = []
    for i in range(0, len(concepts), 50):
         Concepts_str.append(
             "".join(concepts[i:i+50])
         )
else:
    Concepts_str = ["".join(concepts)]

openai.api_key = "xxx" # replace with your API key
prompt = 'Please assign a score between 0 and 1 based on the importance of the following concepts in visually recognizing class "{}". The output should strictly follow this format: <class>, <concept>, <score>. Here are the concepts: \n{}'
os.makedirs("concept_collection/results/GPT/{}".format(gpt_model), exist_ok=True)

with open( 'concept_collection/results/GPT/{}/{}_{}.txt'.format(gpt_model, data_name, concept_source), 'w') as f:
  for i in tqdm(range(len(classes))):
    for concepts_str in Concepts_str:
      cls = classes[i]
      prompt_i = prompt.format(cls, concepts_str)
      # print(cls)
      response = openai.ChatCompletion.create(
          model=gpt_model,
          messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt_i},
          ],
          temperature=0.75,
          max_tokens=1000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
      )
      answer = response.choices[0]["message"]["content"]
      f.write(answer+'\n')
f.close()




