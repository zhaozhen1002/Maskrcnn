import os
import shutil
path = r'C:\Users\pasonatech\Desktop\car_detection\tensorflow\Mask_RCNN\train_data\labelme_json'
json_file = os.listdir(path)
i=1
for file in json_file:
    z=str(i)
    z=z.zfill(6)
    sourceFile = os.path.join(path, (file+'\label.png'))
    targetFile = os.path.join(path+'/%s.png'%z)
    shutil.copy(sourceFile, targetFile)
    i += 1

