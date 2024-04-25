import pandas as pd

data = pd.read_csv("E:\Word\Documents\大三下\计算机工程实训\大作业\data\儿科5-14000.csv",encoding="ANSI")


data_new = data.drop(["department","title"],axis = 1)  # 删除title这列数据



data_new.to_csv("E:\Word\Documents\大三下\计算机工程实训\大作业\data\儿科5-14000_new.csv", index=0,encoding="ANSI")
