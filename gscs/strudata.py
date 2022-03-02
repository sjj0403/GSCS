import pandas as pd 
import javalang
import os
import sys
import re 
import pickle
import numpy as np
from javalang.ast import Node

#javalang解析源代码
def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

#返回node的语义表示
def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
        #print("str: "+node)
    elif isinstance(node, set):
        token = 'Modifier'#node.pop()
        #print("set: "+token)
    elif isinstance(node, Node):
        token = node.__class__.__name__
        #print("node: "+token)
    print(token)
    return token

#返回当前node的孩子节点序列
def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []
    print(children)
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item
    
    return list(expand(children))

#先序遍历AST
def get_sequence(node, sequence):
    token, children = get_token(node), get_children(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)

#根据先序遍历算法返回AST序列
def trans_to_sequences(ast):
    sequence = []
    get_sequence(ast, sequence)
    return sequence

def hump2sub(hump_str):
    seq = []
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p,r'\1_\2', hump_str)#.lower()
    q = re.compile(r'([a-z][a-z])([0-9])')
    newsub = re.sub(q,r'\1_\2', sub)
    #seq = sub.split(r'_d+')
    seq = re.split('_',newsub)
    return seq

def underline2sub(underline_str):
    seq = []
    seq = underline_str.split('_')#.lower().split('_')
    seq = list(filter(None, seq))
    return seq

def get_method_name(node):
    children = get_children(node)
    for item in children:
        if isinstance(item, str):
            return item

#split java9(only)
def get_adj(node, sequence, adj, parent_index, current_index, method_name):
    token, children = get_token(node), get_children(node)
    if len(sequence) >= 200:
        return
    sequence.append(token)
    adj[current_index][current_index] = 1
    if current_index == 0 : 
        pass
    else:        
        adj[parent_index[-1]][current_index] = 1
        parent_index.pop()
    #终端结点
    if len(children) == 0:
        # 切分长字符串        
        if token.startswith('"') and token.endswith('"'):
            token = token.strip('"')
            stringL = token.split(' ')
            if len(stringL) > 1 and len(stringL)+len(sequence) <= 200:
                sequence[-1] = 'stringsplit'
                index = current_index
                for j in range(len(stringL)):
                    if "_" in stringL[j]:
                        sub = underline2sub(stringL[j])
                        if len(sub) > 1 and len(sub) + len(sequence) <= 200:               
                            index = current_index
                            newsub2 = []
                            for k in range(len(sub)):
                                sub2 = hump2sub(sub[k])
                                if len(sub2) > 1:
                                    newsub2.extend(sub2)
                                else:
                                    newsub2.append(sub[k])
                            if len(newsub2) + len(sequence) <= 200:
                                for s in range(len(newsub2)):
                                    sequence.append(newsub2[s])
                                    adj[index][current_index+s+1] = 1
                                    adj[current_index+s+1][current_index+s+1] = 1
                            else:
                                for s in range(len(sub)):
                                    sequence.append(sub[s])
                                    adj[index][current_index+s+1] = 1
                                    adj[current_index+s+1][current_index+s+1] = 1
                        elif len(sub) == 1:
                            sequence[-1] = sub[0]
                            sub2 = hump2sub(sub[0])
                            if len(sub2) > 1 and len(sub2) + len(sequence) <= 200:
                                index = current_index
                                for t in range(len(sub2)):
                                    sequence.append(sub2[t])
                                    adj[index][current_index+t+1] = 1
                                    adj[current_index+t+1][current_index+t+1] = 1

            elif len(stringL) == 1:
                if "_" in stringL[0]:
                    sub = underline2sub(stringL[0])
                    if len(sub) > 1 and len(sub) + len(sequence) <= 200: 
                        sequence[-1] = 'stringsplit'              
                        index = current_index
                        newsub2 = []
                        for k in range(len(sub)):
                            sub2 = hump2sub(sub[k])
                            if len(sub2) > 1:
                                newsub2.extend(sub2)
                            else:
                                newsub2.append(sub[k])
                        if len(newsub2) + len(sequence) <= 200:
                            for s in range(len(newsub2)):
                                sequence.append(newsub2[s])
                                adj[index][current_index+s+1] = 1
                                adj[current_index+s+1][current_index+s+1] = 1
                        else:
                            for s in range(len(sub)):
                                sequence.append(sub[s])
                                adj[index][current_index+s+1] = 1
                                adj[current_index+s+1][current_index+s+1] = 1
                    elif len(sub) == 1:
                        sequence[-1] = sub[0]
                        sub2 = hump2sub(sub[0])
                        if len(sub2) > 1 and len(sub2) + len(sequence) <= 200:
                            sequence[-1] = 'stringsplit'
                            index = current_index
                            for t in range(len(sub2)):
                                sequence.append(sub2[t])
                                adj[index][current_index+t+1] = 1
                                adj[current_index+t+1][current_index+t+1] = 1
            else:
                sequence[-1] = stringL[0]
        
        #切分方法名
        elif token == method_name:
            sub = underline2sub(token)
            if len(sub) > 1 and len(sub) + len(sequence) <= 200:                
                sequence[-1] = 'methodnamesplit'
                index = current_index
                newsub2 = []
                for k in range(len(sub)):
                    sub2 = hump2sub(sub[k])
                    if len(sub2) > 1:
                        newsub2.extend(sub2)
                    else:
                        newsub2.append(sub[k])
                if len(newsub2) + len(sequence) <= 200:
                    for s in range(len(newsub2)):
                        sequence.append(newsub2[s])
                        adj[index][current_index+s+1] = 1
                        adj[current_index+s+1][current_index+s+1] = 1
                else:
                    for s in range(len(sub)):
                        sequence.append(sub[s])
                        adj[index][current_index+s+1] = 1
                        adj[current_index+s+1][current_index+s+1] = 1
            elif len(sub) == 1:
                sequence[-1] = sub[0]
                sub2 = hump2sub(sub[0])
                if len(sub2) > 1 and len(sub2) + len(sequence) <= 200:
                    sequence[-1] = 'methodnamesplit'
                    index = current_index
                    for t in range(len(sub2)):
                        sequence.append(sub2[t])
                        adj[index][current_index+t+1] = 1
                        adj[current_index+t+1][current_index+t+1] = 1
        #切分变量名
        elif "_" in token:
            sub = underline2sub(token)
            if len(sub) > 1 and len(sub) + len(sequence) <= 200:                
                sequence[-1] = 'variablesplit'
                index = current_index
                newsub2 = []
                for k in range(len(sub)):
                    sub2 = hump2sub(sub[k])
                    if len(sub2) > 1:
                        newsub2.extend(sub2)
                    else:
                        newsub2.append(sub[k])
                if len(newsub2) + len(sequence) <= 200:
                    for s in range(len(newsub2)):
                        sequence.append(newsub2[s])
                        adj[index][current_index+s+1] = 1
                        adj[current_index+s+1][current_index+s+1] = 1
                else:
                    for s in range(len(sub)):
                        sequence.append(sub[s])
                        adj[index][current_index+s+1] = 1
                        adj[current_index+s+1][current_index+s+1] = 1
            elif len(sub) == 1:
                sequence[-1] = sub[0]
                sub2 = hump2sub(sub[0])
                if len(sub2) > 1 and len(sub2) + len(sequence) <= 200:
                    sequence[-1] = 'variablesplit'
                    index = current_index
                    for t in range(len(sub2)):
                        sequence.append(sub2[t])
                        adj[index][current_index+t+1] = 1
                        adj[current_index+t+1][current_index+t+1] = 1
        
    for i in range(len(children)):
        parent_index.append(current_index)

    for child in children:
        current_index = len(sequence)
        get_adj(child, sequence, adj, parent_index, current_index, method_name)

#得到切分后的邻接矩阵
def get_adjacent(ast):
    adj = np.zeros((201, 201), dtype=np.int)
    sequence = []
    parent_index = []
    method_name = get_method_name(ast)
    get_adj(ast, sequence, adj, parent_index, 0, method_name)
    return adj

#得到切分后的node序列
def get_adjacent2(ast):
    adj = np.zeros((201, 201), dtype=np.int)
    sequence = []
    parent_index = []
    method_name = get_method_name(ast)
    get_adj(ast, sequence, adj, parent_index, 0, method_name)
    return sequence

# test = 'public int getPushesLowerbound() { return "pushes_Lowerbound"; }'
# tree = parse_program(test)
# parent_index = []
# seqc = get_adjacent(tree)
# ss = get_adjacent2(tree)
# print(seqc)
# print(ss)




