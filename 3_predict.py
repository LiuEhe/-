import tkinter as tk  # Python的标准图形用户界面库，用于创建图形用户界面。
from tkinter import filedialog
from PIL import Image, ImageTk  # Python图像处理库，提供图像处理功能。
import numpy as np
from util import preprocess_image, load, get
import yaml  # 用于读取和解析YAML文件的库。

char_styles = get('char_styles')  # 字体样式
new_size = get('new_size')  # 新尺寸


class ImageClassifierApp:
    def __init__(self, model_path):
        # 使用util.load加载最佳模型
        self.model = load('best_model',model_path)
        #self.model = load('best_model',get("model_root"))

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title('Image Classifier')
        self.root.geometry("400x500")

        # 创建一个按钮用于选择图像
        self.button = tk.Button(self.root, text='选择图像', command=self.select_image)
        self.button.pack()

        # 创建一个标签用于显示图像
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # 创建一个标签用于显示预测的类别
        self.prediction_label = tk.Label(self.root)
        self.prediction_label.pack()

        # 启动GUI事件循环
        self.root.mainloop()

    def select_image(self):
        # 打开文件对话框以选择图像
        image_path = filedialog.askopenfilename()
        #print(image_path)

        # 使用util的preprocess_image函数预处理图像
        # TODO
        preprocess_image(image_path, new_size)
        img_df = preprocess_image(image_path, new_size)

        #print(img_df.shape)

        # 使用加载的最佳模型执行推理
        predicted_class = self.model.predict(img_df.reshape(1,-1))# TODO
        #predicted_class = load('best_model',get("model_root"))
        print(predicted_class)


        # 用PIL读取图像，并设置读取图像后的窗口的大小
        # TODO
        pil_image = Image.open(image_path)
        #print(pil_image)

        # 将PIL图像转换为PhotoImage并更新标签
        image_tk = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

        # 更新预测标签
        self.prediction_label.config(text=f'预测类别: {char_styles[predicted_class[0]]}')
        #self.prediction_label.config(text=f'预测类别: {predicted_class.predict(img_df)}')


# 启动应用程序
app = ImageClassifierApp(f'{get("model_root")}/best_model')
