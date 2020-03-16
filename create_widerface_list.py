import os
import json

label_map = {'face': 1, 'background': 0}

def parse_wider_annotation_file(file, image_root):
    img_paths = []
    objects = []
    with open(file, 'r') as f:
        lines = f.readlines()
    
    face_count = []
    face_loc = []
    flag = False
    count = 0
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            # (xmin, ymin, w, h) -> (xmin, ymin, xmax, ymax)
            loc = [int(line[0]), int(line[1]), int(line[2])+int(line[0]), int(line[3])+int(line[1])]
            face_loc.append(loc)

        if flag:
            face_count.append(int(line))
            flag = False
            count = int(line)
        
        if 'jpg' in line:
            img_paths.append(os.path.join(image_root, line))
            flag = True
    
    total_face = 0
    for k in face_count:
        boxes = []
        for j in range(total_face, total_face + k):
            boxes.append(face_loc[j])
        labels = [1 for i in range(k)]
        objects.append({'face_count':k, 'boxes':boxes, 'labels':labels})
        total_face += k
    
    print("Total images:{}, Total faces:{}".format(len(objects), total_face))

    return img_paths, objects

def creat_data_lists(wider_face_root, output_folder):

    
    # 转为绝对路径
    wider_face_root = os.path.abspath(wider_face_root)
    print(wider_face_root)
    # 训练集和验证集标记文件
    train_list_file = os.path.join(wider_face_root, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
    val_list_file = os.path.join(wider_face_root, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    # 训练集和验证集图像根目录
    wider_train_image = os.path.join(wider_face_root, 'WIDER_train', 'images')
    wider_val_image = os.path.join(wider_face_root, 'WIDER_val', 'images')

    # 训练集
    # 解析训练集标注文件
    train_img_paths, tiran_objects = parse_wider_annotation_file(train_list_file, wider_train_image)

    assert len(train_img_paths) == len(tiran_objects)

    # 保存成json文件
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as f:
        json.dump(train_img_paths, f)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as f:
        json.dump(tiran_objects, f)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)


    # 验证集
    # 解析验证集标注文件
    val_img_paths, val_objects = parse_wider_annotation_file(val_list_file, wider_val_image)
    
    assert len(val_img_paths) == len(val_objects)

    # 保存成json文件
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as f:
        json.dump(val_img_paths, f)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as f:
        json.dump(val_objects, f)



if __name__ == "__main__":
    creat_data_lists('C:\\Users\\caiso\\Desktop\\MyS3FD\\data\\WIDER', './')