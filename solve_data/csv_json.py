import pandas as pd


import json

fr=open("E:\Word\Documents\大三下\计算机工程实训\大作业\data\儿科5-14000_new.csv","r",encoding="ANSI")
ls=[]
for line in fr:
    line=line.replace("\n","")
    ls.append(line.split(","))
fr.close()
fw=open("E:\Word\Documents\大三下\计算机工程实训\大作业\data\儿科5-14000_new.json","w",encoding='utf-8')
for i in range(1,len(ls)):
    ls[i]=dict(zip(ls[0],ls[i]))
b = json.dumps(ls[1:],indent=4,ensure_ascii=False)
print(b)
fw.write(b)
fw.close()

