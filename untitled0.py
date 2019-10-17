# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:45 2019

@author: tomson
"""

import pickle
with open('results.pkl','rb') as file:
    a=pickle.load(file)
from matplotlib import pyplot as plt 
#y1=[item[0] for item in a]
#y2=[item[3] for item in a]
#y3=[item[6] for item in a]
#x=range(1,401)
#plt.plot(x,y1)
#plt.show()
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}
plt.figure(figsize=(6, 4), dpi=100)

movie_name = ['SA','MA','MIA','LV','BERT']
first_day = [0.6, 0.4, 1.4, 1.0, 1.9]
first_weekend = [0.4, 0.5, 1.6, 1.2, 1.3]

# 先得到movie_name长度, 再得到下标组成列表
x = range(len(movie_name))

plt.bar(x, first_day, width=0.2,label='F1 for ECE')
# 向右移动0.2, 柱状条宽度为0.2
plt.bar([i + 0.2 for i in x], first_weekend, width=0.2,label='EM for ED+ECPE')

# 底部汉字移动到两个柱状条中间(本来汉字是在左边蓝色柱状条下面, 向右移动0.1)
plt.xticks([i + 0.1 for i in x], movie_name)
plt.xticks(fontproperties = 'Times New Roman', size = 10)
plt.legend(loc='upper left',prop=font1)
plt.ylabel('Absolute Improvement(%)', fontsize=10,fontdict=font1)
plt.savefig('haha1.eps')
plt.show()
