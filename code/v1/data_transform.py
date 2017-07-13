import numpy as np
import pandas as pd

cd C:\Users\YJ\Desktop
//读取数据
data = pd.read_csv("15_data.csv")
list1 = pd.read_csv("15_normalInfo.csv")
list2 = pd.read_csv("15_failureInfo.csv")
//加一个target属性，正常为1，故障为0，无效数据为-1
data["target"] = np.zeros((data.shape[0], 1)) - 1 //先让其都为无效数据，待会再赋值1和0

//数据复制函数，从正常的时间内读取数据，将其target置为1；故障的时间内读取数据，将target置为0			
for index, element in enumerate(list1["startTime"]):
	data.iloc[data[data["time"] == list1.iloc[index, 0]].index[0] : data[data["time"] == list1.iloc[index, 1]].index[0] + 1,-1] = 1

for index, element in enumerate(list2["startTime"]):
	data.iloc[data[data["time"] == list2.iloc[index, 0]].index[0] : data[data["time"] == list2.iloc[index, 1]].index[0] + 1,-1] = 0
	
//去掉现在target值仍为-1的数据
data = data[data.target != -1]