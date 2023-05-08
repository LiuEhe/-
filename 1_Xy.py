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
import glob

# 1. 读取配置文件中的信息
train_dir = get("train") # 获取 训练数据路径
char_styles = get("char_styles") # 获取 字符样式列表，注意: 必须是列标
new_size = get("new_size") # 获取 新图像大小元组, 注意: 必须包含h和w

#print(train_dir)

# 2. 生成X,y 
print("# 读取训练数据并进行预处理，") 
# TODO

#创建X，y列表
X = []
y = []

#获取符样式列表
char_styles = get("char_styles")  # 获取 字符样式列表，注意: 必须是列标


# 使用glob.glob函数查找符合条件的文件，并将结果保存到image_files列表中
#递归列表
image_files = [glob.glob(f"{train_dir}/train_{category}*") for category in char_styles]


for i in range(5):
    for element in tqdm(image_files[i], desc=f"处理 {char_styles[i]} 图像", unit="bit"):
        label = element.split('_')[1]
        X.append(preprocess_image(element, new_size))
        y.append(char_styles.index(label))

#转换成np数组
X = np.array(X)
y = np.array(y)
#print(y)

#转换成float64类型
X = X.astype(np.float64)

#转换成int64类型
y = y.astype(np.int64)


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
