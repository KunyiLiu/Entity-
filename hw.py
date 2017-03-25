import numpy as np
import pandas as pd
import re as re
# read data
amazon=pd.read_csv("amazon.csv")
rotten_tomatoes=pd.read_csv("rotten_tomatoes.csv",encoding = "ISO-8859-1")
train=pd.read_csv("train.csv")
rotten_selected=rotten_tomatoes.iloc[:,:10]
# deal with time
prob=[]
for index, row in rotten_selected.iterrows():  
    timelong=row['time']
    if str(timelong)!='nan':
        m=re.search(r'(\d+) hr. (\d+) min.',timelong)
        m1=re.search(r'(\d+) hr.',timelong)
        m2=re.search(r'(\d+) min.',timelong)
        if m:
            time_tup=m.groups()
            minute=int(time_tup[0])*60+int(time_tup[1])
        elif m1:
            time_tup=m1.groups()
            minute=int(time_tup[0])*60
        elif m2:
            time_tup=m2.groups()
            minute=int(time_tup[0])  
        else:
            prob.append(index)
        rotten_selected.set_value(index, 'time', minute)
rotten_selected = rotten_selected.drop('year',1)
#combine star1 to star6 into a column of sets star
rotten_selected['star'] = [set() for x in range(len(rotten_selected.index))]
for index, row in rotten_selected.iterrows():
    star=set()
    for i in range(3,9):
        if str(row[i])!='nan':
            star.add(row[i])
    rotten_selected.set_value(index, 'star',star)  
rotten_selected = rotten_selected.iloc[:,[0,1,2,9]]
