#from meteor.meteor import Meteor
from rouge.rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import pandas as pd
import numpy as np
import sys
'''
result_file = '/home/zxq/code/zxq12-CSCGDual-master/CSCGDual1/output.data'

with open(result_file, 'r') as file:
    lines = file.readlines()
    res, gts = {}, {} 
    for i, line in enumerate(lines):
       
        
        if i%2 == 1:
            res[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]
        elif i%2 == 0:
            gts[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]

      
hyps = []
refs = []
bleu_score = 0.0

print(res[1])
print('---------------------')
print(gts[1])


for k in res:
    assert k in gts
    hyps.append(res[k][0])
    refs.append(gts[k][0])
print(hyps[1])
for hyp, ref in zip(hyps, refs):
    hyp = hyp.strip().split()
    ref = ref.strip().split()
    bleu_score += sentence_bleu([ref], hyp, smoothing_function = SmoothingFunction().method4)

print("score_Bleu: ")
print(bleu_score*1.0/len(hyps))

score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
print("Meteor: ")
print(score_Meteor)
score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
print("ROUGe: ")
print(score_Rouge)

'''
#result_file = open('/media/sjj/KINGSTON/result12.9/python_rnn_pointer_noshuffle_best.txt','r')
path = '/media/sjj/KINGSTON/second/result/transformer/java200/epoch9/out.pkl'
source = pd.read_pickle(path)
res, gts = {}, {}        
refs = []  #biaozhun 
hyps = []  #shengcheng
bleu_score = 0.0

refs = source['references'].tolist()
print(refs[0])
hyps = source['predictions'].tolist()
print(hyps[0])
'''
count=1
for line in result_file:
    if count%2 ==1:
        refs.append(line.strip(' \n'))
    else:
        hyps.append(line.strip(' \n'))
    count=count+1
'''
for i in range(len(hyps)):
    res[i]=[hyps[i]]
    gts[i]=[refs[i]]
print(res[0])
print(gts[0])
#print(res)

for hyp, ref in zip(hyps, refs):
    hyp = hyp.strip().split()
    ref = ref.strip().split()
    bleu_score += sentence_bleu([ref], hyp, smoothing_function = SmoothingFunction().method4)

print("score_Bleu: ")
print(bleu_score*1.0/len(hyps))



'''
score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
print("Meteor: ")
print(score_Meteor)
'''
score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
print("ROUGe: ")
print(score_Rouge)
