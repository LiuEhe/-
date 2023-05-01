# 0. 引入必要的包
# TODO
from util import get
from util import dump
import os
from util import preprocess_image
import cv2
#from util import *
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import time

# 1. 读取配置文件中的信息
train_dir = get("train") # 获取 训练数据路径
char_styles = get("char_styles") # 获取 字符样式列表，注意: 必须是列标
new_size = get("new_size") # 获取 新图像大小元组, 注意: 必须包含h和w

#print(train_dir)

# 2. 生成X,y 
print("# 读取训练数据并进行预处理，") 
# TODO


X = []
y = []

#读取图片
for file_name in os.listdir(train_dir):
    img = preprocess_image(train_dir+'/'+ file_name, new_size)
    X.append(img)

#创建映射
#mapping = {'楷书': 1, '篆书': 2, '行书': 3, '草书': 4, '隶书': 5}
mapping = {'草书': 1, '楷书': 2, '隶书': 3, '行书': 4, '篆书': 5}
#mapping = {'1': '楷书', 2:'篆书', 3:'行书', 4:'草书', 5:'隶书'}

#读取标签
for file_name in os.listdir(train_dir):
    if file_name.endswith('.jpg'):
        y.append(file_name.split('_')[1])

X = np.array(X)
y = np.array(y)

X = X.astype(np.float64)

#print(y)
#print(X)
# 使用tqdm创建进度条，并遍历linspace_list中的元素
for i in range(5):
    if i == 0:   
        for element in tqdm(y[0:1000],desc=f'处理  {y[0]}  图像', unit="每秒的单位"):
            time.sleep(0.001)

    if i == 1:   
        for element in tqdm(y[1000:2000],desc=f'处理  {y[1000]}  图像', unit="每秒的单位"):
            time.sleep(0.001)

    if i == 2:   
        for element in tqdm(y[2000:3000],desc=f'处理  {y[2000]}  图像', unit="每秒的单位"):
            time.sleep(0.001)   

    if i == 3:   
        for element in tqdm(y[3000:4000],desc=f'处理  {y[3000]}  图像', unit="每秒的单位"):
            time.sleep(0.001)  

    if i == 4:   
        for element in tqdm(y[4000:5000],desc=f'处理  {y[4000]}  图像', unit="每秒的单位"):
            time.sleep(0.001)           

vectorized_mapper = np.vectorize(lambda x: mapping[x])
y = vectorized_mapper(y).astype(np.int64)

# 3. 分割测试集和训练集
print("# 将数据按 80% 和 20% 的比例分割")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)         #TODO

# 4. 打印样本维度和类型信息
print("X_train: ", X_train.shape, X_train.dtype)  # 训练集特征的维度和类型
print("X_test: ", X_test.shape, X_test.dtype)  # 测试集特征的维度和类型
print("y_train: ", y_train.shape, y_train.dtype)  # 训练集标签的维度和类型
print("y_test: ", y_test.shape, y_test.dtype)  # 测试集标签的维度和类型

# 5. 序列化分割后的训练和测试样本
# TODO

obj = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}

name = 'X_train,X_test,y_train,y_test'
loc = './Xys/Xy'

dump(obj, name , loc)