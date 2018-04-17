#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @PROJECT : jupyter_Projects
# @Time    : 2018/4/13 13:17
# @Author  : Chen Yuelong
# @Mail    : yuelong.chen@oumeng.com.cn
# @File    : utils.py
# @Software: PyCharm

from __future__ import absolute_import, unicode_literals
import sys, os
from itertools import product
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
import re

def get_amino_acid_pattern(*string):
    '''
    通过输入一个氨基酸序列（多肽序列，获取氨基酸pattern），可能是多条序列(多条多肽序列组合成一个蛋白)
    例如，输入 'ATAATTATA'
    表现的pattern一共有 AA,AT,AC,AG,TA,TT,TC,TG,CA,CT,CC,CG,GA,GT,GC,GG及每个碱基可能为开始和结束的标志，
    _A,_T,_C,_G,_A,_T,_C,_G
    返回的为一个矩阵（假设只有两种氨基酸A和T）：
    则ATAATTATA的矩阵如下
    -------------------------------------------
        A_ T_ AA AT TA TT
    _A  0  0  0  1  0  0
    _T  0  0  0  0  0  0
    AA  0  0  0  1  0  0
    AT  0  0  0  0  2  1
    TA  1  0  1  1  0  0
    TT  0  0  0  0  1  0
    --------------------------------------------
    :param string: 氨基酸序列（20个氨基酸缩写组成的string）
    :return:np.array
    '''
    rownames,colnames = get_titles(amion_list())
    pddata = get_dataFrame(rownames,colnames)
    pddata = update_dataFrame(pddata,string)
    return pddata.values



def update_dataFrame(dataFram,strings):
    for string in strings:
        string = '_{}_'.format(string.upper())
        # print(string)
        for i in range(2,len(string)):
            former = '{}{}'.format(string[i-2],string[i-1])
            now = '{}{}'.format(string[i-1],string[i])
            dataFram.loc[former,now] += 1
    return dataFram


def get_dataFrame(rownames,colnames):
    pddata = pd.DataFrame(np.zeros((len(rownames), len(colnames))),
                          index=rownames, columns=colnames)
    return pddata

def get_titles(tmplist):
    '''

    :param tmplist:
    :return:列名，行名
    '''
    pro_list = list(product(tmplist,repeat=2))
    pro_list.sort()
    row_titles = list(map(lambda x:'_{}'.format(x),tmplist))
    list(map(lambda x:row_titles.append(''.join(x)),pro_list))
    col_titles = list(map(lambda x:'{}_'.format(x),tmplist))
    list(map(lambda x: col_titles.append(''.join(x)), pro_list))
    return row_titles,col_titles

def amion_list():
    '''
    返回氨基酸list
    :return:
    '''
    am_li = [
    'A',    #丙氨酸 Ala
    'R',    #精氨酸 Arg
    'D',    #天冬氨酸 Asp
    'C',    #半胱氨酸 Cys
    'Q',    #谷氨酰胺 Gln
    'E',    #谷氨酸 Glu/Gln
    'H',    #组氨酸 His
    'I',    #异亮氨酸 Ile
    'G',    #甘氨酸 Gly
    'N',    #天冬酰胺 Asn
    'L',    #亮氨酸 Leu
    'K',    #赖氨酸 Lys
    'M',    #甲硫氨酸 Met
    'F',    #苯丙氨酸 Phe
    'P',    #脯氨酸 Pro
    'S',    #丝氨酸 Ser
    'T',    #苏氨酸 Thr
    'W',    #色氨酸 Trp
    'Y',    #酪氨酸 Tyr
    'V',    #缬氨酸 Val
    'X',    #非标准蛋白氨基酸
    'B',    #非标准蛋白氨基酸
    'J',    #非标准蛋白氨基酸
    'O',    #非标准蛋白氨基酸
    'U',    #非标准蛋白氨基酸
    'Z'     #非标准蛋白氨基酸
    ]
    # am_li = ['A','T','C','G']
    return am_li

def main():
    '''
    测试流程
    '''
    print(get_amino_acid_pattern('aacatgatgccatgcatgcatgcatgtttatatat'))




if __name__ == '__main__':
    main()