# MaskRCNN-Damage-Detection
## Introduction
This is an Step-by-step tutorial of Mask R-CNN - How to train your own dataset on Python 3, Keras, and TensorFlow.Using [Mask R-CNN](https://arxiv.org/abs/1703.06870) based on https://github.com/matterport/Mask_RCNN.  
このcodeは「CUDA9.1 tensorflow1.12 keras2.1.6 TeslaV100X2」でテストしました。  
  
## 手順
### データの用意と前処理
* 使用ツール：[Labelme](https://github.com/wkentaro/labelme)  
 ```bash
 アノテーションルール：class名_番号
 例：cat_1,cat_2,dog_1
 ```
* train_dataの準備  
 [train_data](train_data)のようにフォルダとデータをご用意ください。  
  <img width="420" height="157" src=figures/1.PNG/>  
  1.pic(学習写真)      
  <div align=center><img width="600" height="380" src=figures/pic.PNG/></div>

  2.json(labelmeで作ったjsonファイル)  　　  
  <div align=center><img width="600" height="310" src=figures/json.PNG/></div>  
  
  3.labelme_json(labelmeのlabelme_json_to_datasetというscriptで作れます)  
    [labelme_json_to_datasetの使い方：](https://github.com/wkentaro/labelme/issues/420)  　　
  <div align=center><img width="600" height="310" src=figures/labelme_json.PNG/></div>   
    folder中身  
  <div align=center><img width="600" height="150" src=figures/detail.PNG/></div> 
  
  4.cv2_mask(folder中のlabel.pngをcv2_maskにcopyしてください)  
  <div align=center><img width="600" height="310" src=figures/cv2_mask.PNG/></div>  
  


## train.py/inference.pyの編集  
* [demo.ipynb](samples/demo.ipynb)の編集　
 

