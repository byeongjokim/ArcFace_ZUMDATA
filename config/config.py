class Config(object):
    backbone = 'resnet50'
    
    num_classes = 200
    metric = 'arc_margin'
    easy_margin = False
    use_se = False

    if backbone == "resnet18":
        input_shape = (1, 128, 128)
    else:
        input_shape = (3, 112, 112)

    optimizer = 'sgd'

    inf_root = "/"
    inf_json = "/home/kbj/projects/ZUMDATA/inf/people/OBS"

    root = "/home/kbj/projects/ZUMDATA/total/"
    total_list = "/home/kbj/projects/ZUMDATA/total.txt"
    train_list = "/home/kbj/projects/ZUMDATA/train.txt"
    val_list = "/home/kbj/projects/ZUMDATA/val.txt"
    test_list = "/home/kbj/projects/ZUMDATA/test.txt"

    zum_test_list = '/home/kbj/projects/ZUMDATA/test_pair.txt'

    checkpoints_path = 'checkpoints'
    test_model_path = 'checkpoints/resnet50_40.pth'    

    max_epoch = 200
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    train_batch_size = 64  # batch size
    test_batch_size = 32

    use_gpu = True  # use GPU or not
    gpu_id = '1'
    num_workers = 4  # how many workers for loading data

    save_interval = 10
    print_freq = 100  # print info every N batch