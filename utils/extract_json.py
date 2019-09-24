import os
import shutil
path = '/Users/xuzhongwei/Source/ai/dataset/labelme_json'
json_file = os.listdir(path)
i=1
for file in json_file:
    z=str(i)
    z=z.zfill(6)
    print(file)
    if os.path.isdir(path + "/" + file):
        sourceFile = os.path.join(path + ('/' + file+'/label.png'))
        targetFile = os.path.dirname(os.path.dirname(path + "/" + file)) + '/cv2_mask/%s.png'%file[:6]

        shutil.copy(sourceFile, targetFile)
        i += 1

