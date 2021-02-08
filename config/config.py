class Config(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 619
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    train_root = '/home/kbj/projects/similar_celeb/ZUM/'
    train_file_list = '/home/kbj/projects/similar_celeb/ZUM/train.txt'
    
    val_root = '/home/kbj/projects/similar_celeb/ZUM/'
    val_file_list = '/home/kbj/projects/similar_celeb/ZUM/train.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    
    save_interval = 10
    eval_interval = 10

    batch_size = 64 

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
