from mtcnn.mtcnn import MTCNN
import AlignDlib

import cv2
import random
from itertools import combinations

import argparse
import glob
import json
import os
    
def xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x, y, x+w, y+h]

def parsing_data(data_root, data_json):
    print("[+] parsing json")
    
    assert os.path.isfile(data_json)
    assert os.path.isdir(data_root)
    
    with open(data_json) as f:
        json_data = json.load(f)
    
    data = []
    for ann in json_data["annotation"]["frames"]:
        if not ann["people"]:
            continue
            
        frame = ann["filepath"]
        people = ann["people"]
        
        file_path = data_root + frame
        if not os.path.isfile(file_path):
            continue
        
        data += [
            [file_path, xywh2xyxy(person["bbox"]), person["category_id"]]
            for person in people
        ]
    
    classes = sorted(list(set([d[2] for d in data])))
    print("[+] parsed {} data {} classes".format(len(data), len(classes)))
            
    return data, classes

def align(align_model, face, img):
    feature = face['keypoints']

    points = []
    points.append(feature['left_eye'])
    points.append(feature['right_eye'])
    mouth_x = round((feature['mouth_left'][0] + feature['mouth_right'][0])/2)
    mouth_y = round((feature['mouth_left'][1] + feature['mouth_right'][1])/2)
    points.append([mouth_x, mouth_y])
    landmarks = list(map(lambda p: (p[0], p[1]), points))

    out = align_model.align(256, img, #bb=box,
                            landmarks=landmarks,
                            skipMulti=True)
    resized_dim = (128, 128)
    out = cv2.resize(out, resized_dim)

    return out

def crop(data, classes, data_root, total_list, train_list, val_list, test_list, pad=10):
    if not os.path.isdir(data_root):
        os.mkdir(data_root)
        print("[+] mkdir "+ data_root)

    detector = MTCNN()
    align_model = AlignDlib.AlignDlib('')

    total_txt = open(total_list, "w")
    train_txt = open(train_list, "w")
    val_txt = open(val_list, "w")
    test_txt = open(test_list, "w")
    
    no_face_img = []
    
    celebs = {}
    face_num = 0

    for d in data:
        img_file = d[0]
        x1, y1, x2, y2 = d[1]
        
        label = classes.index(d[2])

        img = cv2.imread(img_file)
        
        height, width, _ = img.shape

        x2 = int(min(width-1, x2+pad))
        y2 = int(min(height-1, y2+pad))
        x1 = int(max(0, x1-pad))
        y1 = int(max(0, y1-pad))

        img = img[y1:y2, x1:x2]
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(img_file, (x1, y1, x2, y2))
            no_face_img.append(img_file)
            continue
        
        faces = detector.detect_faces(rgb_img)
        faces = [f for f in faces if f['confidence'] > 0.9]
        
        new_file = os.path.join(data_root, str(face_num).zfill(10)+".png")            
        
        if len(faces) >= 1:
            face = faces[0]
            aligned_face = align(align_model, face, img)
            cv2.imwrite(new_file, aligned_face)    
        else:
            no_face_img.append(face_num)
            continue
        
        face_num += 1
        face_file = new_file.split("/")[-1]
        total_txt.write("{} {}\n".format(face_file, str(label)))
        try:
            celebs[label].append(face_file)
        except:
            celebs[label] = [face_file]
            
    print("no face in {} images".format(str(len(no_face_img))))
    print(no_face_img)

    train_celebs = {}
    val_celebs = {}
    test_celebs = {}
    for i in celebs:
        random.shuffle(celebs[i])
        train_celebs[i] = celebs[i][:-6]
        val_celebs[i] = celebs[i][-6:-5]
        test_celebs[i] = celebs[i][-5:]
    
    for i in train_celebs:
        for row in train_celebs[i]:
            train_txt.write("{} {}\n".format(row, str(i)))

    for i in val_celebs:
        for row in val_celebs[i]:
            val_txt.write("{} {}\n".format(row, str(i)))
    
    for i in test_celebs:
        for row in test_celebs[i]:
            test_txt.write("{} {}\n".format(row, str(i)))

    total_txt.close()
    train_txt.close()
    val_txt.close()
    test_txt.close()
    
    print("[+] cropped {} faces".format(face_num))
    
def make_pair_list(identity_list, pair_list, same_num, diff_num, total_same_num, total_diff_num):
    # make pair list using identitiy_list
    # img1 1
    # img2 2
    # img3 1
    # ->
    # img1 img2 0
    # img1 img3 1

    with open(identity_list, "r") as f:
        data = f.readlines()
    
    images = []
    labels = []
    
    check = {}
    for d in data:
        img, label = d.split()

        if not label in check:
            check[label] = {"files": [], "same": 0, "diff": 0}

        check[label]["files"].append(img)

        images.append(img)
        labels.append(label)
    
    same_pairs = []
    # same labels
    for i in check:
        candi = list(combinations(check[i]["files"], 2))
        
        candi = [[cnd[0], cnd[1], 1] for cnd in candi]

        random.shuffle(candi)

        candi = candi[:same_num]
        
        same_pairs += candi
    
    try:
        same_pairs = random.sample(same_pairs, total_same_num)
    except:
        pass
    
    diff_pairs = []
    # diff labels
    for img, label in zip(images, labels):
        cnt = 0
        while cnt != diff_num:
            idx = random.randrange(0, len(images))
            if not label == labels[idx]:
                diff_pairs.append([img, images[idx], 0])
                cnt += 1
    
    try:
        diff_pairs = random.sample(diff_pairs, total_diff_num)
    except:
        pass
    
    pairs = same_pairs + diff_pairs
    
    with open(pair_list, "w") as f:
        for pair in pairs:
            f.write("{} {} {}\n".format(pair[0], pair[1], pair[2]))

def balance_data(data, classes, num=20):
    balanced_data = []
    
    classes_dict = {i:[] for i in classes}
    for i, d in enumerate(data):
        classes_dict[d[2]].append(i)
        
    for i in classes_dict:
        try:
            inds = random.sample(classes_dict[i], num)
            balanced_data += [data[ind] for ind in inds]
        except:
            pass
    
    return balanced_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--num_classes', default=200, type=int)
    parser.add_argument('--metric', default="arc_margin")
    parser.add_argument('--easy_margin', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    parser.add_argument('--optimizer', default="sgd")
    
    parser.add_argument('--data_root', default="/")
    parser.add_argument('--data_json', default="/home/kbj/projects/ZUMDATA/inf/people/OBS")
    
    parser.add_argument('--root', default="/home/kbj/projects/ZUMDATA/total")
    
    parser.add_argument('--total_list', default="/home/kbj/projects/ZUMDATA/total.txt")
    parser.add_argument('--train_list', default="/home/kbj/projects/ZUMDATA/train.txt")
    parser.add_argument('--val_list', default="/home/kbj/projects/ZUMDATA/val.txt")
    parser.add_argument('--test_list', default="/home/kbj/projects/ZUMDATA/test.txt")
    parser.add_argument('--zum_test_list', default="/home/kbj/projects/ZUMDATA/pair.txt")
    
    parser.add_argument('--checkpoints_path', default="checkpoints")
    
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--lr_step', default=10, type=int)
    parser.add_argument('--lr_decay', default=0.2, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--gpu_id', default="1")
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--save_interval', default=10, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    
    opt = parser.parse_args()
    
    if opt.backbone == "resnet18":
        opt.input_shape = (1, 128, 128)
        opt.test_model_path = 'checkpoints/resnet18/resnet18_30.pth'
    else:
        opt.input_shape = (3, 112, 112)
        opt.test_model_path = 'checkpoints/resnet50/resnet50_50.pth'

    data, classes = parsing_data(opt.data_root, opt.data_json)

    balanced_data = balance_data(data, classes)
    balanced_class = list(set([d[2] for d in balanced_data]))
    
    crop(balanced_data, balanced_class, opt.root, opt.total_list, opt.train_list, opt.val_list, opt.test_list)

    make_pair_list(opt.test_list, opt.zum_test_list, 4, 2, 300, 300)
