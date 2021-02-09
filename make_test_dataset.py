import random
from itertools import combinations

def make_val_list(identity_list, pair_list, same_num=3, diff_num=2):
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
    make_val_list("val.txt", "pairs.txt")