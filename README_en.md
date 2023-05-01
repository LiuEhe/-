## Project Description
This project is a calligraphy font style recognizer. By inputting images, it can recognize the calligraphy font style in the images. The project includes the following files:
- 0_setting.yaml: Configuration file, including the list of calligraphy font styles, target size for resizing images, etc. 
- 1_Xy.py: Preprocess images and generate training and testing datasets. 
- 2_fit.py: Use LazyClassifier to evaluate multiple classification models and select the model with the highest F1 score and save it.
- 3_predict.py: Create a simple graphical user interface where users can select an image and the program will display the predicted calligraphy font style.   
- util.py: Contains some auxiliary functions such as image preprocessing, saving and loading files, etc.
## Project Run Effect Screenshots 
<img src="https://github.com/LiuEhe/Calligraphy-Style-Recognition/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE/1.jpg" width="150" height="187.5"><img src="https://github.com/LiuEhe/Calligraphy-Style-Recognition/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE/2.jpg" width="150" height="187.5"><img src="https://github.com/LiuEhe/Calligraphy-Style-Recognition/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE/3.jpg" width="150" height="187.5"><img src="https://github.com/LiuEhe/Calligraphy-Style-Recognition/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE/4.jpg" width="150" height="187.5"><img src="https://github.com/LiuEhe/Calligraphy-Style-Recognition/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE/5.jpg" width="150" height="187.5">
## Functions
1. Preprocess images and generate training and testing datasets. 
2. Use LazyClassifier to evaluate multiple classification models and select the model with the highest F1 score and save it.
3. Create a simple graphical user interface where users can select an image and the program will display the predicted calligraphy font style.
## Instructions 
  This project uses traditional machine learning methods, and the prediction accuracy hovers around 0.7. For learning reference only.
## Dependencies 
- Python 
- Scikit-learn 
- LazyPredict 
- OpenCV
- PIL
- Tkinter
- PyYAML 
## Usage 
1. Make sure all dependent libraries have been installed. 
2. Run 1_Xy.py to generate training and testing datasets. 
3. Run 2_fit.py to evaluate multiple classification models and save the best model. 
4. Run 3_predict.py to start the graphical user interface and select an image for prediction.
## Notes
- Please click [notion](https://liuehe.notion.site/79d73daae145425e9c513dee39b10d84) and select 1. Calligraphy Style Recognition to download the data. 
- The serialized data exceeds 100MB, the file is not uploaded, please run 1_Xy.py to get it, or go to [notion](https://liuehe.notion.site/79d73daae145425e9c513dee39b10d84) to download.
- Please generate related folders and place files according to the settings in the configuration file 0_setting.yaml. 
- Make sure all dependent libraries have been installed. （已编辑） 
