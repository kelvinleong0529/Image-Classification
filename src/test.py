import pandas as pd

df = pd.read_csv("E:/Machine Learning/project2/input/face-classifier/Mutli_Label_dataset/list_attr_celeba.txt",sep =" ",nrows=100)
temp = df.drop(["Id","Young"],axis = 1)
print(temp)