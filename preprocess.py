from config import Config
from mtcnn.mtcnn import MTCNN
import AlignDlib

import cv2
import random
from itertools import combinations

import glob
import json

def parsing(origin_root, origin_json):
    jpg_files = glob.glob(os.path.join(origin_root, "*/*.jpg"))
    json_files = glob.glob(os.path.join(origin_json, "*/*.json"))

    jpg_data = {i.split("/")[-1]:i for i in jpg_files}

    class_set = []
    data = []
    for json_file in json_files:
        with open(json_file) as f:
            json_data = json.load(f)
        
        filename = json_data["images"][0]["filename"]
        anns = json_data["annotations"]

        class_set += [ann["class"] for ann in anns]

        data += [[jpg_data[filename], ann["box"], ann["class"]] for ann in anns]

    return data, list(set(class_set))

def crop(data, classes, data_root, total_list, pad=10):

    detector = MTCNN()
    align_model = AlignDlib.AlignDlib('')

    txt = open(total_list, "w")

    idx = 0
    # crop and save and make list
    for d in data:
        img_file = d[0]
        x, y, w, h = d[1]

        img = cv2.imread(img_file)
        
        width, height, _ = rgb_img.shape

        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        label = classes.index(d[2])

        img = img[x1:x2, y1:y2]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = detector.detect_faces(rgb_img)
        faces = [f for f in faces if f['confidence'] > 0.9]
        
        if len(faces) == 1:
            face = faces[0]

            aligned_face = align(face, img)

            cv2.imwrite(os.path.join(data_root, str(idx).zfill(10) + ".jpg"), aligned_face)    
            txt.write("{} {}\n".format(str(idx).zfill(10) + ".jpg", str(label)))

            idx += 1 
    
    txt.close()

def align(face, img):
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

def make_list(total_list, train_list, val_list)

def make_pair_list(identity_list, pair_list, same_num, diff_num):
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
    
    pairs = []
    # same labels
    for i in check:
        candi = list(combinations(check[i]["files"], 2))
        
        candi = [[cnd[0], cnd[1], 1] for cnd in candi]

        random.shuffle(candi)

        candi = candi[:same_num]
        
        pairs += candi

    # diff labels
    for img, label in zip(images, labels):
        cnt = 0
        while cnt != diff_num:
            idx = random.randrange(0, len(images))
            if not label == labels[idx]:
                pairs.append([img, images[idx], 0])
                cnt += 1
    
    with open(pair_list, "w") as f:
        for pair in pairs:
            f.write("{} {} {}\n".format(pair[0], pair[1], pair[2]))



if __name__ == '__main__':
    opt = Config()

    data, classes = parsing(opt.origin_root, opt.origin_json)
    crop(data, classes, opt.root, opt.total_list):

    make_list(opt.total_list, opt.train_list, opt.val_list)
    make_pair_list(opt.val_list, opt.pair_list, opt.same_num, opt.diff_num)
