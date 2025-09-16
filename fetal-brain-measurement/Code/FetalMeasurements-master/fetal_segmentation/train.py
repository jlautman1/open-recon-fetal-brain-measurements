import logging
from pathlib import Path
from sacred import Experiment
import training
import training.train_functions.metrics
import training.model
from data_generation.generator import get_training_and_validation_generators
from training.train_functions.training import *
from utils.read_write_data import save_data_splits, save_norm_params, save_old_model
import uuid
import socket

"""
Training with sacred framework. By defult the hardcoded configuration is used. If you want to use existing configuration just add as argument
with [config_path]
"""
#initializing sacred
my_path = os.path.abspath(os.path.dirname(__file__))
sacred_path = os.path.join(my_path, '../runs/')
ex = Experiment(sacred_path)

from sacred.observers import FileStorageObserver
from data_generation.data_preparation import *

@ex.config
def config():
    my_path = os.path.abspath(os.path.dirname(__file__))

    ##############################################################
    #data parameters

    overwrite = False #Overwrite creates new data splits, use it carefully

    scans_dir = os.path.join(my_path, '../../../data/fetal_mr/body/FIESTA/') #directory of the raw scans
    data_dir = os.path.join(my_path, '../data/body_FIESTA/')

    normalization = {
        0: False,
        1: 'all',
        2: 'each'
    }[1]  # Normalize by all or each data mean and std
    all_modalities = ["volume"]
    training_modalities = all_modalities
    ext = ""  # add ".gz" extension if needed

    data_split_validation = 0.9
    data_split_test = 0.18
    split_dir = os.path.join(data_dir, 'debug_split')

    training_file = os.path.join(split_dir, "training_ids.txt")
    validation_file = os.path.join(split_dir, "validation_ids.txt")
    test_file = os.path.join(split_dir, "test_ids.txt")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(split_dir).mkdir(parents=True, exist_ok=True)
    ###############################################################
    #training params

    batch_size = 3
    patches_per_epoch = 800

    labels = (1,)
    n_labels = len(labels)

    is_3D = True  # Enable for 3D Models
    if is_3D:
        # Model params (3D)
        patch_shape = (96, 96)  # switch to None to train on the whole image
        patch_depth = 64
        model_name = 'isensee'  # or 'unet'
        truth_index = 0
        truth_size = 64
    else:
        #Model params (2D) - should increase "batch_size" and "patches_per_epoch"
        patch_shape = (96, 96)  # switch to None to train on the whole image
        patch_depth = 5
        model_name = 'unet' # or 'isensee'
        truth_index = 2
        truth_size = 1

    # choose model
    chosen_model = {
        '3D': {
            'unet': 'unet_model_3d',
            'isensee': 'isensee2017_model_3d'
        },
        '2D': {
            'unet': 'unet_model_2d',
            'isensee': 'isensee2017_model'
        }
    }['3D' if is_3D else '2D'][model_name]

    input_shape = tuple(list(patch_shape) + [patch_depth])

    if is_3D:
        input_shape = [1] + list(input_shape)

    augment = {
        # 'flip': [0.5, 0.5, 0.5],  # augments the data by randomly flipping an axis during
        # 'permute': False,  # NOT SUPPORTED (data shape must be a cube. Augments the data by permuting in various directions)
        # 'translate': (15, 15, 7),  #
        # 'scale': (0.1, 0.1, 0),  # i.e 0.20 for 20%, std of scaling factor, switch to None if you want no distortion
        # # "iso_scale": {
        # #     "max": 1
        # # },
        # 'rotate': (0, 0, 90),  # std of angle rotation, switch to None if you want no rotation
        # 'min_crop_size': (70,70,15), #minimum size of cropped volume around ground truth
        # 'poisson_noise': 0.5,
        #  "contrast": {
        #      'min_factor': 0.3,
        #      'max_factor': 0.7
        #  },
        # "piecewise_affine": {
        #     'scale': 2
        # },
        # "elastic_transform": {
        #     'alpha': 5,
        #     'sigma': 10
        # },
        # "intensity_multiplication_range": [0.2,1.8],
        # "coarse_dropout": {
        #     "rate": 0.2,
        #     "size_percent": [0.10, 0.30],
        #     "per_channel": True
        # }
    }
    preprocess = 'window_1_99'
    scale = None
    truth_crop = False
    categorical = False
    n_epochs = 300  # cutoff the training after this many epochs
    patience = 3  # learning rate will be reduced after this many epochs if the validation loss is not improving
    early_stop = 50  # training will be stopped after this many epochs without the validation loss improving
    initial_learning_rate = 5e-4
    learning_rate_drop = 0.5  # factor by which the learning rate will be reduced
    validation_split = 0.90  # portion of the data that will be used for training %
    dropout_rate=0
    old_model_path=None

    skip_blank_train = False  # if True, then patches without any target will be skipped
    skip_blank_val= False  # if True, then patches without any target will be skipped
    drop_easy_patches_train = True  # will randomly prefer balanced patches (50% 1, 50% 0)
    drop_easy_patches_val = False  # will randomly prefer balanced patches (50% 1, 50% 0)
    loss = {
        0: 'binary_crossentropy_loss',
        1: 'dice_coefficient_loss',
        2: 'focal_loss',
        3: 'dice_and_xent',
        4: 'dice_distance_weighted_loss'
    }[1]

@ex.main
def my_main():

    train_generator, validation_generator, n_train_steps, n_validation_steps, data_file_opened = generate_data()

    train_data(train_generator, validation_generator, n_train_steps, n_validation_steps)

    data_file_opened.close()

@ex.capture
def train_data(train_generator, validation_generator, n_train_steps, n_validation_steps, overwrite,
            loss, chosen_model, input_shape, initial_learning_rate, dropout_rate, n_epochs, learning_rate_drop,
            patience, early_stop, training_file, validation_file, test_file, data_dir, old_model_path):

    save_data_splits(ex.observers[0].dir, training_file, validation_file, test_file)
    save_norm_params(ex.observers[0].dir, data_dir)
    if(old_model_path != None):
        save_old_model(ex.observers[0].dir, old_model_path)

    model_file = os.path.join(ex.observers[0].dir, 'epoch_')

    if not overwrite and len(glob.glob(model_file + '*.hdf5')) > 0:
        model_path = get_last_model_path(model_file)
        print('Loading model from: {}'.format(model_path))
        model = load_old_model(model_path)
    else:
        # instantiate new model
        loss_func = getattr(training.train_functions.metrics, loss)
        model_func = getattr(training.model, chosen_model)
        model = model_func(input_shape,
                           initial_learning_rate=initial_learning_rate,
                           **{'dropout_rate': dropout_rate,
                              'loss_function': loss_func})
    model.summary()

    # run training
    train_model(model=model,
                ex=ex,
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=initial_learning_rate,
                learning_rate_drop=learning_rate_drop,
                learning_rate_patience=patience,
                early_stopping_patience=early_stop,
                n_epochs=n_epochs,
                output_folder=ex.path)

@ex.capture
def generate_data(scans_dir, data_dir, normalization, training_modalities, ext, overwrite, batch_size,data_split_test, data_split_validation, validation_file, training_file, test_file,
                  labels, n_labels, patch_shape, patch_depth, augment, skip_blank_train, skip_blank_val, drop_easy_patches_train, drop_easy_patches_val,
                  patches_per_epoch, is_3D, truth_index, truth_size, truth_crop, categorical, preprocess, scale):

    data_file_opened = create_load_hdf5(normalization=normalization, data_dir=data_dir, scans_dir=scans_dir, train_modalities=training_modalities, ext=ext,
                                        overwrite=overwrite, preprocess=preprocess, scale=scale)

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=batch_size,
        data_split_test=data_split_test,
        data_split_validation = data_split_validation,
        overwrite=overwrite,
        validation_keys_file=validation_file,
        training_keys_file=training_file,
        test_keys_file=test_file,
        n_labels=n_labels,
        labels=labels,
        patch_shape=(*patch_shape, patch_depth),
        validation_batch_size=batch_size,
        augment=augment,
        skip_blank_train=skip_blank_train,
        skip_blank_val=skip_blank_val,
        patches_per_epoch=patches_per_epoch,
        is3d=is_3D,
        drop_easy_patches_train=drop_easy_patches_train,
        drop_easy_patches_val=drop_easy_patches_val,
        truth_index=truth_index,
        truth_size=truth_size,
        truth_crop=truth_crop,
        categorical=categorical)

    return train_generator, validation_generator, n_train_steps, n_validation_steps, data_file_opened

if __name__ == '__main__':

    log_dir = '../log/'
    log_level = logging.INFO
    my_path = os.path.abspath(os.path.dirname(__file__))
    log_path = os.path.join(my_path, log_dir)

    # uid = uuid.uuid4().hex
    # fs_observer = FileStorageObserver.create(os.path.join(log_path, uid))
    fs_observer = FileStorageObserver.create(log_path)

    ex.observers.append(fs_observer)

    # initialize logger
    logger = logging.getLogger()
    hdlr = logging.FileHandler(os.path.join(ex.observers[0].basedir, 'messages.log'))
    FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(filename)20s:%(lineno)-5s %(funcName)-25s %(message)s"
    formatter = logging.Formatter(FORMAT)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    # logger.removeHandler(lhStdout)
    logger.setLevel(log_level)
    ex.logger = logger
    logging.info('Experiment {}, run {} initialized'.format(ex.path, ex.current_run))

    ex.run_commandline()