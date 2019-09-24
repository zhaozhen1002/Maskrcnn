import os

path = "../rail_train_data/json"

json_files = os.listdir(path)

for file in json_files:

    sourceFilePath = os.path.join(path, file)
    root, ext = os.path.splitext(sourceFilePath)

    if ext == ".json":
        # s = "./labelme_json_to_dataset " + path+file
        # os.system(s)
        os.system("./labelme_json_to_dataset %s"%(path+file))