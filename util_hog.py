# 引入必要的库
import cv2
import time
import joblib 
import os
import yaml

# 分别用于加载数据，颜色空间转换，特征计算和图像的对比度处理
from skimage import data, color, feature, exposure

# TODO

#读取yaml文件
with open("0_setting.yaml", "r",encoding='utf-8') as f:
    config = yaml.safe_load(f) # config就自动包含了yaml文件中的所有字典信息

#print(config)

# 获取 0_setting.yaml 中的键 key 对应的值 value
def get(key):
    # TODO
    value = config[key]
    return value

# 预处理图像, 把图像设置为指定大小之后，展平返回
def preprocess_image(file_name, new_size):
    # 1. 读取图像灰度图
    # TODO
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE,flags=cv2.IMREAD_COLOR)
    
    # 2. 调整图像大小为 new_size
    # TODO
    img = cv2.resize(img, new_size,interpolation=cv2.INTER_AREA)

    
    # 3. 计算HOG特征，orientations为方向的数量，pixels_per_cell定义了每个单元格的大小，cells_per_block定义了每个块的大小，visualize=True表示需要返回HOG图像
    # 注意：这里为了更加清晰的展示HOG图像，选择了16*16的小窗大小，正常情况下，为了获取更高的精细程度，可以选择8*8的小窗大小
    hog_vec, hog_edge = feature.hog(img, orientations=9, 
                                pixels_per_cell=(8, 8), 
                                cells_per_block=(2, 2), 
                                visualize=True)

    img = hog_edge
    
    # 4. 将图像展平为一维数组
    # TODO
    img =img.ravel()
    return img

# 用joblib把叫做 name 的对象 obj 保存(序列化)到位置 loc
def dump(obj,name, loc):
    start = time.time()
    print(f"把{name}保存到{loc}") 
    # TODO 此处序列化对象
    joblib.dump(obj,loc)

    end = time.time()
    print(f"保存完毕,文件位置:{loc}, 大小:{os.path.getsize(loc) / 1024 / 1024:.3f}M")
    print(f"运行时间:{end - start:.3f}秒")

# 用joblib读取(反序列化)位置loc的对象obj,对象名为name
def load(name, loc):
    print(f"从{loc}提取文件{name}")
    #TODO 此处反序列化对象
    obj = joblib.load(loc)
    return obj