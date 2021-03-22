from config import Config
from mtcnn.mtcnn import MTCNN
import AlignDlib

import cv2
import random
from itertools import combinations

import glob
import json
import os

def matching_inf(filename):
    path = "/data/video/vod_tag01/bai/99/OBS"
    path_2fs = "/data/video/vod_tag02/OBS_p0_v0~v49/Images"
    
    program, video, frame = filename.split("/")[-3:]
    
    if "_" in frame:
        _, _, _, frame = frame.split("_")
        
    assert len(program) == 12
    assert len(video) == 10
    assert len(frame) == 15
    
    list_2fps = [
        "video-"+str(i).zfill(4) for i in range(50)
    ]
    
    if program == "program-0000" and video in list_2fps:
        path = os.path.join(path_2fs, video, frame[:-5]+".png")
    else:
        path = os.path.join(path, program, "frames", video, frame[:-5]+".png")
    
    if os.path.isfile(path):
        return path
    else:
        print(path)
        return False
    
def xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x, y, x+w, y+h]

def parsing_inf(inf_root, inf_json):
    print("infinfinfinfinfinfinfinfinf")
    json_files = glob.glob(os.path.join(inf_json, "*/*/*.json"))
    data = []
    for json_file in json_files:
        with open(json_file) as f:
            json_data = json.load(f)

        filename = json_data["images"][0]["filename"]
        anns = json_data["annotations"]
        
        try:
            data += [[matching_inf(json_file), xywh2xyxy(ann["box"]), ann["class"]] for ann in anns if matching_inf(json_file) and ann["class"] != "person"]
        except KeyError:
            print('KeyError in ' + json_file)
            
    return data, list(set([d[2] for d in data]))

# def parsing_mind(origin_root, mind_json):
#     print("mindmindmindmindmindmindmind")
#     json_files = glob.glob(os.path.join(mind_json, "*/*.json"))
#     data = []
#     for json_file in json_files:
#         vname = json_file.split("/")[-1].split(".")[0]
#         with open(json_file) as f:
#             json_data = json.load(f)
        
#         images = {i["id"]:i["file_name"] for i in json_data["images"]}
#         face_cls = {i["id"]:i["name"] for i in json_data["categories"]["face"]}
        
#         data += [
#             [
#                 os.path.join(origin_root, "program-0007", "frames", vname, images[ann["image_id"]]),
#                 xywh2xyxy(ann["bbox"]),
#                 face_cls[ann["face_id"]]
#             ] for ann in json_data["annotations"]["face"]
#             if os.path.isfile(os.path.join(origin_root, "program-0007", "frames", vname, images[ann["image_id"]]))
#         ]
#     return data, list(set([d[2] for d in data]))

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
            # cv2.imwrite(new_file, rgb_img)
            no_face_img.append(face_num)
            continue
        
        total_txt.write("{} {}\n".format(new_file, str(label)))
        face_num += 1

        try:
            celebs[label].append(new_file)
        except:
            celebs[label] = [new_file]
            
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
    
    same_pairs = random.sample(same_pairs, total_same_num)
    
    diff_pairs = []
    # diff labels
    for img, label in zip(images, labels):
        cnt = 0
        while cnt != diff_num:
            idx = random.randrange(0, len(images))
            if not label == labels[idx]:
                diff_pairs.append([img, images[idx], 0])
                cnt += 1
    
    diff_pairs = random.sample(diff_pairs, total_diff_num)
    
    pairs = same_pairs + diff_pairs
    
    with open(pair_list, "w") as f:
        for pair in pairs:
            f.write("{} {} {}\n".format(pair[0], pair[1], pair[2]))

def balance_data(data, num=20):
    balanced_data = []
    
    for i in classes_dict:
        try:
            inds = random.sample(classes_dict[i], num)
            balanced_data += [data[ind] for ind in inds]
        except:
            pass
    
    return balanced_data

if __name__ == '__main__':
    opt = Config()

    data, classes = parsing_inf(opt.inf_root, opt.inf_json)
    # data2, classes2 = parsing_mind(opt.mind_origin_root, opt.mind_json)
    
    # data = data1 + data2
    # classes = list(set(classes1 + classes2))
    print(len(data), len(classes))

    classes_dict = {i:[] for i in classes}
    for i, d in enumerate(data):
        classes_dict[d[2]].append(i)

    balanced_data = balance_data(data)
    balanced_class = list(set([d[2] for d in balanced_data]))
    print(len(balanced_data), len(balanced_class))
    
    crop(data, classes, opt.root, opt.total_list, opt.train_list, opt.val_list, opt.test_list):

    make_pair_list(opt.test_list, opt.zum_test_list, 5, 5, 300, 300)
