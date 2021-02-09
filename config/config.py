class Config(object):
    num_classes = 619
    easy_margin = False
    use_se = False
    optimizer = 'sgd'

    origin_root = "/home/kbj/projects/arcface-pytorch/origin/"
    origin_json = "/home/kbj/projects/arcface-pytorch/origin/"

    root = '/home/kbj/projects/arcface-pytorch/ZUM/'
    train_list = '/home/kbj/projects/arcface-pytorch/ZUM/train.txt'
    val_list = '/home/kbj/projects/arcface-pytorch/ZUM/val.txt'
    pair_list = '/home/kbj/projects/arcface-pytorch/ZUM/pair.txt'

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
    

