import os
import pandas as pd
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
    concepts = [c.strip() for c in concepts]
f.close()
with open("asset/class_names/{}.txt".format(data_name), "r") as f:
    classes = f.readlines()
    classes = [c.strip() for c in classes]
f.close()
print("{} concepts and {} classes".format(len(concepts), len(classes)))


with open('concept_collection/results/GPT/{}/{}_{}.txt'.format(gpt_model, data_name, concept_source), 'r') as f:
    Answers = f.readlines()
f.close()

unformatted_lines = []
unformatted_lines_2 = []
index_DF = 0
DF = pd.DataFrame({})
for i in range(len(Answers)):
    answer = Answers[i]
    if answer.endswith('\\\n'):
        answer = answer[:-2]
    elif answer.endswith('\n'):
        answer = answer.split('\n')[0]
    if '. ' in answer:
        answer =  answer.split('. ')[1]
    if answer.startswith('- '):
        answer =  answer[2:]
    if answer.startswith('**') and answer.endswith('**'):
        answer = answer[2:-2]
    try:
        answer_list = answer.split(', ')
        assert len(answer_list) >= 3
    except:
        unformatted_lines_2.append(answer+'\n')
        print("Error 2:", answer)
    try:
        if len(answer_list) == 3:
            cls, concept, weight = answer_list
        else:
            cls, concept, weight = answer_list[0], answer_list[1:-1], answer_list[-1]
            concept = ', '.join(concept)
        # print(cls, concept, weight)
        if cls.startswith('"') and cls.endswith('"'):
            cls = cls[1:-1]
        if cls.startswith("'") and cls.endswith("'"):
            cls = cls[1:-1]
        if cls.startswith("**") and cls.endswith("**"):
            cls = cls[2:-2]
        if cls.startswith("*") and cls.endswith("*"):
            cls = cls[1:-1]
        if cls.startswith("<") and cls.endswith(">"):
            cls = cls[1:-1]
        if cls.startswith('class "'):
            cls = cls[7:-1]
        if cls.startswith('class '):
            cls = cls[6:]
        if cls == 'class':
            cls = last_cls

        if concept.startswith('"') and concept.endswith('"'):
            concept = concept[1:-1]
        if concept.startswith("'") and concept.endswith("'"):
            concept = concept[1:-1]
        if concept.startswith("**") and concept.endswith("**"):
            concept = concept[2:-2]
        if concept.startswith("*") and concept.endswith("*"):
            concept = concept[1:-1] 
        if concept.startswith("<") and concept.endswith(">"):
            concept = concept[1:-1]

        if weight.startswith('"') and weight.endswith('"'):
            weight = weight[1:-1]
        if weight.startswith("'") and weight.endswith("'"):
            weight = weight[1:-1]
        if weight.startswith("**") and weight.endswith("**"):
            weight = weight[2:-2]
        if weight.startswith("*") and weight.endswith("*"):
            weight = weight[1:-1]
        if weight.startswith("<") and weight.endswith(">"):
            weight = weight[1:-1]
 
                           
        
        # print(cls, concept, weight)

        assert cls in classes
        assert concept in concepts
        if concept.startswith('"') and concept.endswith('"'):
            print(concept)
        last_cls = cls
        last_concept = concept
        if float(weight)>=0.5:
            DF.loc[index_DF, 'class'] = cls
            DF.loc[index_DF, 'concept'] = concept
            DF.loc[index_DF, 'weight'] = float(weight)
            index_DF+=1
    except:
        
        try:
            if float(weight)>=0.5:
                unformatted_lines.append("{}, {}, {}\n".format(cls, concept, weight))
                print('Error 1:', "{}, {}, {}".format(cls, concept, weight))
        except:
            unformatted_lines_2.append(answer+'\n')
            print("Error 2:", answer)

        


with open ('concept_collection/results/GPT/{}/{}_{}_unformatted_lines_1.txt'.format(gpt_model, data_name, concept_source), 'w') as f:
    for line in unformatted_lines:
        f.write(line)
with open ('concept_collection/results/GPT/{}/{}_{}_unformatted_lines_2.txt'.format(gpt_model, data_name, concept_source), 'w') as f:
    for line in unformatted_lines_2:
        f.write(line)

DF.to_csv('concept_collection/results/GPT/{}/{}_{}.csv'.format(gpt_model, data_name, concept_source), index = False)







