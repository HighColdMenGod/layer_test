# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:18:22 2020

@author: Eric Lehman
"""

o1 = 'ev1_test_article_ids.txt'
o2 = 'ev1_train_article_ids.txt'
o3 = 'ev1_validation_article_ids.txt'     

f1 = 'ev2_test_article_ids.txt'
f2 = 'ev2_train_article_ids.txt'
f3 = 'ev2_validation_article_ids.txt'

def combine(f1, f2, n):
    with open(f1) as tmp:
        txt1 = list(filter(lambda x: x != '', tmp.read().split('\n')))
        
    with open(f2) as tmp:
        txt2 = list(filter(lambda x: x != '', tmp.read().split('\n')))
        
    txt = txt1 + txt2
    with open(n, 'w') as tmp:
        tmp.write('\n'.join(txt))      
        
combine(o1, f1, 'test_article_ids.txt')
combine(o2, f2, 'train_article_ids.txt')
combine(o3, f3, 'validation_article_ids.txt')