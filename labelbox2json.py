## input: .json file(generated from labelbox export)
## output: one json file for one image

## Variable to be modified:
## synetset_path (line13) - classes.txt that contains all the string of classes on every line
## file (line22) - generated single json output file from labelbox
## filePath (line40) - path to your image file
## outputpath (line93) - output annotation file path

import os
import cv2
import json
import numpy as np

count = 0
synset_path = '/home/minshiu/aws-lauretta-sagemaker-XRAY/classes.txt'
with open(synset_path, 'r') as f:
    lines = f.readlines()
classes = [l[:-1] for l in lines]

number = np.arange(len(classes))
reference_list = dict(zip(classes, number))

file = '/home/minshiu/Downloads/export-2019-04-08T08_44_06.612Z.json'
with open(file, 'r') as f:
    lines = f.readlines()

raw_string = lines[0].split('"ID"')
del raw_string[0]
splitted = []

for idx, l in enumerate(raw_string):
    s = '''{"ID"''' + l[:-2]
    if int(idx) == len(raw_string) - 1:
        s += '}'
    splitted.append(json.loads(s))

for idx, item in enumerate(splitted):
    
    file = item.pop('External ID')
    filePath = '/home/minshiu/aws-lauretta-sagemaker-XRAY/' + file
    img = cv2.imread(filePath)
    
    if img is None:
        #print('Skipping file: {} \nReason: image not found'.format(file))
        continue
    
    print('Found image with annotation: {}'.format(file))
    
    (height, width, depth) = img.shape
    item['image_size'] = [{'width':width, 'height':height, 'depth':depth}]
    item['file'] = file
    ann_ls = []
    categories = []
    if item['Label'] != 'Skip':
        lbl = item['Label']
        for k in list(lbl.keys()):
            dct = lbl[k]
            for d in dct:
                ann = {}
                cls_name = list(d.values())[0]
                large_y = max(t['y'] for t in d['geometry'])
                small_y = min(t['y'] for t in d['geometry'])
                large_x = max(t['x'] for t in d['geometry'])
                small_x = min(t['x'] for t in d['geometry'])
                ann['left'] = small_x
                ann['top'] = small_y
                ann['width'] = large_x - small_x
                ann['height'] = large_y - small_y
                if type(cls_name) != list:
                    if cls_name not in reference_list.keys():
                        reference_list[cls_name] = str(len(reference_list))
                    ann['class_id'] = reference_list[cls_name]
                else:
                    if k not in reference_list.keys():
                        reference_list[k] = str(len(reference_list))
                    ann['class_id'] = reference_list[k]
                ann_ls.append(ann)

        item["annotations"] = ann_ls
        x = [ids['class_id'] for ids in ann_ls]
        unique = np.unique(x)
        for u in unique:
            k = [key for key in reference_list.items() if key[1] == str(u)][0][0]
            categories.append({"class_id": u, "name": k})
        item["categories"] = categories
    count += 1
    print(item, '\n')
    print('------------------------Done {} files.------------------------------'.format(count))
    
    keys = list(item.keys())[-4:]
    content = dict(zip([k for k in keys], [item[k] for k in keys]))
    jsonFile = item['file'].split('.')[0] + '.json'
    with open(os.path.join('/home/minshiu/aws-lauretta-sagemaker-XRAY/generated/', jsonFile), 'w') as p:
        json.dump(content, p)
