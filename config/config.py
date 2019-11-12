import os
import pathlib


class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 100

    metric = 'arc_margin'
    easy_margin = False
    # use_se = True
    use_se = False
    # loss = 'focal_loss'
    loss = 'cross_entropy'

    display = True
    # display = False
    finetune = True
    # finetune = False

    train_root = 'data/Datasets/invoices/train/'
    train_list = 'data/Datasets/invoices/train/train_filenames.txt'
    val_list = 'data/Datasets/invoices/train/val_filenames.txt'

    test_root = 'data/Datasets/invoices/test/'
    test_list = 'data/Datasets/invoices/test/test_filenames.txt'

    lfw_root = 'data/Datasets/lfw/lfw-align-128'
    lfw_test_list = 'data/Datasets/lfw/lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18_40.pth'
    test_model_path = 'checkpoints/resnet18_40.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 16

    input_shape = (1, 256, 256)

    # optimizer = 'sgd'
    optimizer = 'Adam'

    use_gpu = True  # use GPU or not
    # use_gpu = False  # use GPU or not
    gpu_id = '0, 1'
    # gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 10  # print info every N batch

    debug_file = 'tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    def __init__(self, root_path):
        print(root_path)
        self.train_root = os.path.join(root_path, self.train_root)
        self.train_list = os.path.join(root_path, self.train_list)
        self.val_list = os.path.join(root_path, self.val_list)
        self.test_root = os.path.join(root_path, self.test_root)
        self.test_list = os.path.join(root_path, self.test_list)
        self.debug_file = os.path.join(root_path, self.debug_file)
        self.lfw_root = os.path.join(root_path, self.lfw_root)
        self.lfw_test_list = os.path.join(root_path, self.lfw_test_list)
        self.checkpoints_path= os.path.join(root_path, self.checkpoints_path)
        self.num_classes = len([dir_name for dir_name in os.listdir(self.train_root)
                                if pathlib.Path(self.train_root).joinpath(dir_name).is_dir()])
