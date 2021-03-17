class Config(object):
    num_classes = 200
    easy_margin = False
    use_se = False
    optimizer = 'sgd'

    # origin_root = "/home/kbj/projects/arcface-pytorch/origin/"
    mind_origin_root = "/data/video/vod_tag02/bai/99/EBS"
    inf_json = "/home/kbj/projects/ZUMDATA/inf/people/OBS"
    mind_json = "./mind"

    root = '/home/kbj/projects/ZUMDATA/total/'
    total_list = '/home/kbj/projects/ZUMDATA/total.txt'
    train_list = '/home/kbj/projects/ZUMDATA/train.txt'
    val_list = '/home/kbj/projects/ZUMDATA/val.txt'
    pair_list = '/home/kbj/projects/ZUMDATA/test_pair.txt'

    val_per_class = 5
    same_num = 3
    diff_num = 2
    
    input_shape = (128, 128)

    checkpoints_path = 'checkpoints'
    
    save_interval = 10
    eval_interval = 10
    print_freq = 100

    batch_size = 64
    val_batch_size = 32
    max_epoch = 200

    lr = 1e-1 
    lr_step = 10
    weight_decay = 5e-4

    gpu_id = "1"
    num_workers = 4
    

