import os
import shutil
import csv

data_path = "./data/train"
csv_path = "./data/trainLabels.csv"

label_dic = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

with open(csv_path, "r") as f:
    lines = f.readlines()

lines = lines[1:]

for line in lines:
    tokens = line.split(",")
    file_name = tokens[0]
    label_name = tokens[1][:-1]
    image_path = os.path.join(data_path, file_name+".png")

    dir_path = os.path.join(data_path, str(label_dic[label_name]))

    shutil.copy(image_path, dir_path)

# dir_name_list = os.listdir(data_path)
# for dir_name in dir_name_list:
#     dir_path = os.path.join(data_path, dir_name)
#     file_name_list = os.listdir(dir_path)
#     print("\tNumber of files in %s = %d" % (dir_name, len(file_name_list)))
#     for file_name in file_name_list:
#         file_path = os.path.join(dir_path, file_name)
