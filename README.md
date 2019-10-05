# Mask R-CNN for train your own dataset
## Introduction
This is an Step-by-step tutorial of Mask R-CNN - How to train your own dataset on Python 3, Keras, and TensorFlow.Using [Mask R-CNN](https://arxiv.org/abs/1703.06870) based on https://github.com/matterport/Mask_RCNN.  
このcodeは「CUDA9.1 tensorflow1.12 keras2.1.6 TeslaV100X2」でテストしました。   

## 手順
### データの用意と前処理
<details>
    <summary>View Detail</summary>

* 使用ツール：[Labelme](https://github.com/wkentaro/labelme)  
 ```bash
 アノテーションルール：class名_番号  
 例：cat_1,cat_2,dog_1
 ```
* train_dataの準備  
 [train_data](train_data)のようにフォルダとデータをご用意ください。  
  <img width="420" height="157" src=figure/1.PNG/>  
  1.pic(学習写真)      
  <div align=center><img width="600" height="380" src=figure/pic.PNG/></div>
  2.json(labelmeで作ったjsonファイル)  　　  
  <div align=center><img width="600" height="310" src=figure/json.PNG/></div>  

  3.labelme_json(labelmeのlabelme_json_to_datasetというscriptで作れます)  
    [labelme_json_to_datasetの使い方](https://github.com/wkentaro/labelme/issues/420)   　　
  <div align=center><img width="600" height="310" src=figure/labelme_json.PNG/></div>   
    folder中身  
  <div align=center><img width="600" height="150" src=figure/detail.PNG/></div> 

  4.cv2_mask(folder中のlabel.pngをcv2_maskにcopyしてください)  
  <div align=center><img width="600" height="310" src=figure/cv2_mask.PNG/></div>  


 </details>


## 学習
* [train.py](samples/shapes/train.py)  
 ```bash
 python train.py --CLASS_NAME ['ラベル1','ラベル2'] --epoch 100 ......
 ```  
 そのほかのパラメータは[config.py](/mrcnn/config.py)にご調整くだせい。
## 推論
* [test.py](samples/shapes/test.py)
 ```bash
 python test.py --class_names ['BG', 'ラベル1', 'ラベル2'] ......
 ```  
 推論結果は[result](/test_data/)に保存します。  
 例：threshsoldを0.5設定すると、threshsold＞0.5の写真のみ保存。
## tensorboardで可視化  
 ```bash
tensorboard --logdir 自分のpath/MaskRCNN-Damage-Detection/logs/shapes20190930T1107 --host 0.0.0.0
http://localhost:6006/
 ```
 <div align=center><img width="800" height="350" src=figure/loss.PNG/></div>
