# 引入必要的库
import cv2
import time
import joblib 
import os
import yaml
import numpy as np

# 分别用于加载数据，颜色空间转换，特征计算和图像的对比度处理
from skimage import  feature

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.utils import load_img, img_to_array

#读取yaml文件
with open("0_setting.yaml", "r",encoding='utf-8') as f:
    config = yaml.safe_load(f) # config就自动包含了yaml文件中的所有字典信息

#print(config)

# 获取 0_setting.yaml 中的键 key 对应的值 value
def get(key):
    value = config[key]
    return value

# 预处理图像, 把图像设置为指定大小之后，展平返回
def preprocess_image(file_name, new_size_vgg):

    # 1. 使用keras内置的读图程序，以224x224的尺寸读取图像文件，结果为一个PIL图像对象
    img = load_img(file_name, target_size=new_size_vgg)

    # 2. 将PIL图像对象转换为NumPy数组
    img = img_to_array(img)

    # 3. 把单幅图像放到一个数组中，虽然只有一幅图像，但是我们仍然需要扩展数组的维度，以适应VGG16模型的输入尺寸要求（模型要求输入为4D张量）
    X = np.array([img])
    # 有的时候你会看到这样的写法: x = np.expand_dims(x, axis=0) 它与以上的代码是一个意思

    # 4. 使用VGG16模型的预处理函数对图像进行预处理，该步骤包括颜色空间的转换、缩放等
    X = preprocess_input(X)

    # 5. # 加载预训练的VGG16模型，不包括顶部的全连接层（include_top=False），因为我们的目标是提取特征，而不是进行分类
    # weights='imagenet' 表示使用在 ImageNet 数据集上预训练的权重，这些权重可以帮助我们更好地提取特征
    # pooling="max" 表示使用最大池化来池化特征图，这可以帮助我们更好地保留特征信息，并且对结果进行大幅度降维
    model = VGG16(weights='imagenet', include_top=False,pooling="max")

    # 6. 使用 VGG16 模型对图像进行特征提取，model.predict(X) 返回一个包含特征向量的数组
    # [0] 表示我们只提取第一张图像的特征向量，因为我们只输入了一张图像
    y = model.predict(X)[0]

    img = y # 展平特征向量，以适应SVM的输入要求
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