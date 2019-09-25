# Mask R-CNN for train your own dataset
## Introduction
This is an Step-by-step tutorial of Mask R-CNN - How to train your own dataset on Python 3, Keras, and TensorFlow.Using [Mask R-CNN](https://arxiv.org/abs/1703.06870) based on https://github.com/matterport/Mask_RCNN.  
  
  
![Car scrach](figures/title.png)
## 手順
### データの用意
* googleの写真を利用したい場合、[google_images_download](https://github.com/hardikvasa/google-images-download)と推薦します。  
 ```bash
 pip install google_images_download  
 googleimagesdownload -k "キーワード" -l 100
 ```
### 学習データアノテーションと処理
* 使用ツール：[Labelme](https://github.com/wkentaro/labelme)  
 ```bash
 # python3
 conda create --name=labelme python=3.6  
 source activate labelme  
 labelme
 ```
* アノテーションルール  
![ano_rule](figures/labelme_rule.png)
* train_dataの準備  
 [train_data](train_data)のようにフォルダとデータをご用意ください。  
  <img width="420" height="157" src=figures/train_data_format.png/>  
  pic    
  <div align=center><img width="600" height="550" src=figures/pic.png/></div>

  json   
  <div align=center><img width="600" height="550" src=figures/json.png/></div>  
  
  cv2_mask  
  <div align=center><img width="600" height="550" src=figures/cv2_mask.png/></div>  
  
  labelme_json  
  <div align=center><img width="600" height="550" src=figures/labelme_json.png/></div>  
  labelme2dataset  
  <div align=center><img width="600" height="180" src=figures/labelme2dataset.png/></div> 
  


## train.py/inference.pyの編集  
* [demo.ipynb](samples/demo.ipynb)の編集 
 

