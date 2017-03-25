import numpy as np
import pandas as pd
import re as re
# read data
amazon=pd.read_csv("amazon.csv")
rotten_tomatoes=pd.read_csv("rotten_tomatoes.csv",encoding = "ISO-8859-1")
train=pd.read_csv("train.csv")
rotten_selected=rotten_tomatoes.iloc[:,:10]
## amazon data processing
# deal with time period
amazon = amazon.drop('cost', 1)
for i in range(amazon.shape[0]):
    starcol = str(amazon.loc[i]['star']) 
    if bool(re.search(r'\d',starcol)):
        amazon.set_value(i,'time',starcol)
        amazon.set_value(i,'star',np.nan)
    # cope with time column
    timelong = str(amazon.loc[i]['time'])
    if timelong.find('/') != -1:
        amazon.set_value(i, 'time', np.nan)
    else:        
        h = int(timelong.find('hour'))
        mins = int(timelong.find('minute'))
        comma = int(timelong.find(','))
        if h != -1:
            hours = int(timelong[h-2])
            minutes = int(timelong[comma+2:mins-1])
        elif mins != -1:
            hours = 0
            minutes = int(timelong[:mins-1])
        else:
            amazon.set_value(i, 'time',np.nan)
        amazon.set_value(i, 'time', hours*60 + minutes)
# deal with star names
for i in range(amazon.shape[0]):
    # deal with star
    starcol = str(amazon.loc[i]['star'])
    if starcol.find('/')==-1:
        starcol = set(starcol.split(','))
    else:
        starcol = set(starcol.split('/'))
    amazon.set_value(i, 'star',starcol)
    
## rotten tomatoes data cleansing
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
