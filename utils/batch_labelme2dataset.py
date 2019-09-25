import os
path = r'C:\Users\pasonatech\Desktop\car_copy\test/'
json_file = os.listdir(path)
for file in json_file:
    os.system(r'python C:\Users\pasonatech\Anaconda3\envs\labelme\Scripts\labelme_json_to_dataset.exe %s'%(path +file))
