import os
os.environ['PATH'] += ';C:\Program Files\cudnn\cudnn-11.2\cuda/bin/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import datetime
import voxelmorph as vxm
import distutils.dir_util
import pandas as pd
from tensorflow.keras.models import save_model

from NetworkBasis import config as cfg
import NetworkBasis.util as util

import evaluation

import NetworkBasis.loadsavenii as loadsave
from buildmodel import buildmodel
from datagenerator import DataGenerator, get_test_images
import processing

pretrained_model_name = 'weights.h5'


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

logs_path = os.path.join(cfg.path_result, cfg.experiment_name)

distutils.dir_util.mkpath(logs_path)

def training(f, architecture, variant, losses,  loss_weights, learning_rate, nb_features,  nb_epochs, batch_size, seed=42):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training = True

    traindata_files_fixed = pd.read_csv(cfg.train_fixed_csv, dtype=object).values
    traindata_files_moving = pd.read_csv(cfg.train_moving_csv, dtype=object).values

    valdata_files_fixed = pd.read_csv(cfg.vald_fixed_csv, dtype=object).values
    valdata_files_moving = pd.read_csv(cfg.vald_moving_csv, dtype=object).values

    inshape = (cfg.height,cfg.width,cfg.numb_slices)

    vxm_model=buildmodel(architecture, variant, inshape, nb_features)

    vxm_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)

    training_batch_generator = DataGenerator(traindata_files_fixed, cfg.path_fixed, traindata_files_moving, cfg.path_moving, batch_size, inshape,architecture, variant=variant)
    validation_batch_generator = DataGenerator(valdata_files_fixed, cfg.path_fixed, valdata_files_moving, cfg.path_moving, batch_size, inshape,architecture, variant=variant)

    log_dir = logs_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    #Different callbacks
    callback_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                              restore_best_weights=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=logs_path + 'checkpoint',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss', mode='min',
                                                                   save_best_only=True)

    callback = [callback_earlystopping, model_checkpoint_callback, callback_tensorboard]

    validation_steps = cfg.number_of_vald // batch_size
    vxm_model.fit(training_batch_generator, validation_data=validation_batch_generator,
                         validation_steps=validation_steps,
                         epochs=nb_epochs, batch_size=batch_size,
                         callbacks=[callback], verbose=2)

    vxm_model.save_weights(logs_path + str(f) + '/weights.h5')
    try:
        save_model(vxm_model, logs_path + str(f) + "/" )
    except:
        print("model save failed")


def pretrained(f, architecture, variant,losses, loss_weights, learning_rate, nb_features, nb_epochs, steps_per_epoch, batch_size, seed=42):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training = True

    traindata_files_fixed = pd.read_csv(cfg.train_fixed_csv, dtype=object).values
    traindata_files_moving = pd.read_csv(cfg.train_moving_csv, dtype=object).values

    valdata_files_fixed = pd.read_csv(cfg.vald_fixed_csv, dtype=object).values
    valdata_files_moving = pd.read_csv(cfg.vald_moving_csv, dtype=object).values

    inshape = (cfg.height, cfg.width, cfg.numb_slices)

    vxm_model = buildmodel(architecture, variant, inshape, nb_features)

    pretrained_model = "all"
    if pretrained_model=="all":
        vxm_model.load_weights(cfg.path_pretrained + str(f) + "/" + pretrained_model_name, by_name=True)
    elif pretrained_model == "rigid":
        nb_features_0 = [8, 16, 32, 64, 128]
        pretrained_model_0 = buildmodel('adapted_Guetal', 'rigid', inshape, nb_features_0, write_summary=False)
        pretrained_model_0.load_weights(cfg.path_pretrained + str(f) + "/" + pretrained_model_name)
    elif pretrained_model == "rigid_affine":
        nb_features_0 = [[[], [512, 256, 128, 64, 32, 6]],  # rigid
                         [[], [512, 256, 128, 64, 32, 12]]  # affine
                         ]
        pretrained_model_0 = buildmodel('multistage', 'ra', inshape, nb_features_0, write_summary=False)
        pretrained_model_0.load_weights(cfg.path_pretrained + "0/" + pretrained_model_name)
    elif pretrained_model == "affine":
        nb_features_0 = [8, 16, 32, 64, 128]
        pretrained_model_0 = buildmodel('adapted_Guetal', 'affine', inshape, nb_features_0, write_summary=False)
        pretrained_model_0.load_weights(cfg.path_pretrained + str(f) + "/" + pretrained_model_name)

    vxm_model.trainable = True

    # Fine-tune from this layer onwards
    #fine_tune_at = 3 #freeze subnetworks
    #print(vxm_model.layers[fine_tune_at])

    # Freeze all the layers before the `fine_tune_at` layer
    #for layer in vxm_model.layers[:fine_tune_at]:
    #    layer.trainable = False

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)

    if pretrained_model == "rigid":
        vxm_model.get_layer('Guetal_rigid').set_weights(pretrained_model_0.get_weights())
    elif pretrained_model == "rigid_affine":
        vxm_model.get_layer('Guetal_rigid').set_weights(pretrained_model_0.get_layer('Guetal_rigid').get_weights())
        vxm_model.get_layer('Guetal_affine').set_weights(pretrained_model_0.get_layer('Guetal_affine').get_weights())
    if pretrained_model == "affine":
        vxm_model.get_layer('Guetal_affine').set_weights(pretrained_model_0.get_weights())

    training_batch_generator = DataGenerator(traindata_files_fixed, cfg.path_fixed, traindata_files_moving, cfg.path_moving,
                                batch_size, inshape, architecture, variant=variant)
    validation_batch_generator = DataGenerator(valdata_files_fixed, cfg.path_fixed, valdata_files_moving, cfg.path_moving,
                                batch_size, inshape, architecture, variant=variant)

    log_dir = logs_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callback_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                              restore_best_weights=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=logs_path + 'checkpoint',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss', mode='min',
                                                                   save_best_only=True)

    callback = [callback_earlystopping, model_checkpoint_callback, callback_tensorboard]

    validation_steps = cfg.number_of_vald // batch_size

    vxm_model.fit(training_batch_generator, validation_data=validation_batch_generator,
                             validation_steps=validation_steps,
                             epochs=nb_epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size,
                             callbacks=[callback], verbose=2)

    vxm_model.trainable = True

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses,
                      loss_weights=loss_weights)

    vxm_model.save_weights(logs_path + str(f) + '/weights.h5')
    try:
        save_model(vxm_model, logs_path + str(f) + "/")
    except:
        print("model save failed")

def apply(f, architecture, variant, nb_features, batch_size,seed=42):
    '''!
    predict images, (segmentations, ) displacementfields for test files
    use shape fixed image
    '''
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training=False
    cfg.output_disp_all=False

    test_data_fixed = pd.read_csv(cfg.test_fixed_csv, dtype=object).values
    test_data_moving = pd.read_csv(cfg.test_moving_csv, dtype=object).values
    if cfg.seg_available:
        test_data_seg_moving = pd.read_csv(cfg.test_moving_seg_csv, dtype=object).values

    inshape = (cfg.height,cfg.width,cfg.numb_slices)
    vxm_model = buildmodel(architecture, variant, inshape, nb_features)

    print("Test file size: ", len(test_data_fixed))
    vxm_model.load_weights(logs_path + str(f) + '/weights.h5')

    predict_path = logs_path+'predict/'
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    for i in range(int(len(test_data_fixed) / batch_size)):
        test_images = get_test_images(test_data_fixed[i][0], cfg.path_fixed, test_data_moving[i][0], cfg.path_moving,
                                      batch_size, inshape, architecture)
        predictions = vxm_model.predict(test_images, steps=1,
                                        batch_size=batch_size, verbose=2)

        filename_fixed = test_data_fixed[i][0][2:-1]
        filename_moving = test_data_moving[i][0][2:-1]

        cfg.orig_filepath=cfg.path_fixed+filename_fixed

        if cfg.seg_available:
            filename_seg_moving = test_data_seg_moving[i][0][2:-1]

        loadsave.save_pred_disp([predictions[0], predictions[-1]], predict_path, filename_moving)
        processing.warp_img(predict_path, filename_fixed, filename_moving)

        if cfg.seg_available:
            processing.warp_seg(predict_path + "/seg/", filename_fixed, filename_seg_moving, filename_moving=filename_moving)

def evaluate(f):
    '''!
    evaluate predicted images with metrics
    used shape fixed image
    '''

    np.random.seed(42)

    cfg.training = False

    test_data_fixed = pd.read_csv(cfg.test_fixed_csv, dtype=object).values
    test_data_moving = pd.read_csv(cfg.test_moving_csv, dtype=object).values

    if cfg.seg_available:

        test_data_seg_fixed = pd.read_csv(cfg.test_fixed_seg_csv, dtype=object).values
        test_data_seg_moving = pd.read_csv(cfg.test_moving_seg_csv, dtype=object).values

    distutils.dir_util.mkpath(logs_path + 'eval/')
    eval_file_path = logs_path+'eval/' + 'eval-'+str(f)+'.csv'

    header_row = evaluation.make_csv_header()
    util.make_csv_file(eval_file_path, header_row)

    predict_path = logs_path + 'predict/'

    for i in range(len(test_data_fixed)):
        filename_fixed = test_data_fixed[i][0][2:-1]
        filename_moving = test_data_moving[i][0][2:-1]

        if cfg.seg_available:
            filename_seg_fixed = test_data_seg_fixed[i][0][2:-1]
            filename_seg_moving = test_data_seg_moving[i][0][2:-1]
        try:
            result_metrics = {}
            result_metrics['FILENAME_FIXED'] = filename_fixed
            result_metrics['FILENAME_MOVING'] = filename_moving

            if cfg.seg_available:
                result_metrics = evaluation.evaluate_prediction(result_metrics, predict_path,
                                                            ('moved' + '_' + filename_moving),
                                                            cfg.path_fixed_resized, filename_fixed, filename_moving,
                                                            filename_seg_fixed, filename_seg_moving)
            else:
                result_metrics = evaluation.evaluate_prediction(result_metrics, predict_path,
                                                            ('moved' + '_' + filename_moving),
                                                            cfg.path_fixed_resized, filename_fixed, filename_moving)
            #print(result_metrics)
            util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)
            print('Finished Evaluation for ' , filename_fixed , 'and' , filename_moving)
        except RuntimeError as err:
            print("    !!! Evaluation of " , filename_fixed , 'and' , filename_moving , ' failed',err)


    #read csv
    header=pd.read_csv(eval_file_path, dtype=object,sep=';')
    header = header.columns.values
    values = pd.read_csv(eval_file_path, dtype=object,sep=';').values
    np_values = np.empty(values.shape)

    result_metrics['FILENAME_FIXED'] = 'min'
    result_metrics['FILENAME_MOVING'] = ' '

    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
            # print(np_values[j, i + 1])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.min(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

    result_metrics['FILENAME_FIXED'] = 'mean'
    result_metrics['FILENAME_MOVING'] = ' '
    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
            # print(np_values[j, i + 1])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.average(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

    result_metrics['FILENAME_FIXED'] = 'max'
    result_metrics['FILENAME_MOVING'] = ' '
    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
            # print(np_values[j, i + 1])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.max(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

def experiment_1(data_fixed, data_moving, architecture, variant, losses, loss_weights, learning_rate, nb_features, nb_epochs,
                 batch_size, is_training, is_pretrained, is_apply, is_evaluate):
    k_fold=5
    np.random.seed(42)
    all_indices = np.random.permutation(range(0, len(data_fixed)))
    test_folds = np.array_split(all_indices, k_fold)

    if cfg.seg_available:
        data_seg_fixed, data_seg_moving = loadsave.getdatalist_from_csv(cfg.fixed_seg_csv, cfg.moving_seg_csv)

    for f in range(k_fold):
        test_indices = test_folds[f]
        remaining_indices = np.random.permutation(np.setdiff1d(all_indices, test_folds[f]))
        vald_indices = remaining_indices[:cfg.number_of_vald]
        train_indices = remaining_indices[cfg.number_of_vald:]

        train_files_fixed=np.empty(len(train_indices), dtype = "S70")
        train_files_moving=np.empty(len(train_indices), dtype = "S70")
        vald_files_fixed=np.empty(len(vald_indices), dtype = "S70")
        vald_files_moving=np.empty(len(vald_indices), dtype = "S70")
        test_files_fixed=np.empty(len(test_indices), dtype = "S70")
        test_files_moving=np.empty(len(test_indices), dtype = "S70")

        if cfg.seg_available:
            train_files_seg_fixed = np.empty(len(train_indices), dtype="S70")
            train_files_seg_moving = np.empty(len(train_indices), dtype="S70")
            vald_files_seg_fixed = np.empty(len(vald_indices), dtype="S70")
            vald_files_seg_moving = np.empty(len(vald_indices), dtype="S70")
            test_files_seg_fixed = np.empty(len(test_indices), dtype="S70")
            test_files_seg_moving = np.empty(len(test_indices), dtype="S70")

        for i in range(len(train_indices)):
            train_files_fixed[i] = data_fixed[train_indices[i]]
            train_files_moving[i] = data_moving[train_indices[i]]
            if cfg.seg_available:
                train_files_seg_fixed[i] = data_seg_fixed[train_indices[i]]
                train_files_seg_moving[i] = data_seg_moving[train_indices[i]]

        for i in range(len(vald_indices)):
            vald_files_fixed[i] = data_fixed[vald_indices[i]]
            vald_files_moving[i] = data_moving[vald_indices[i]]
            if cfg.seg_available:
                vald_files_seg_fixed[i] = data_seg_fixed[vald_indices[i]]
                vald_files_seg_moving[i] = data_seg_moving[vald_indices[i]]

        for i in range(len(test_indices)):
            test_files_fixed[i] = data_fixed[test_indices[i]]
            test_files_moving[i] = data_moving[test_indices[i]]
            if cfg.seg_available:
                test_files_seg_fixed[i] = data_seg_fixed[test_indices[i]]
                test_files_seg_moving[i] = data_seg_moving[test_indices[i]]

        np.savetxt(cfg.train_fixed_csv, train_files_fixed, fmt='%s', header='path')
        np.savetxt(cfg.vald_fixed_csv, vald_files_fixed, fmt='%s', header='path')
        np.savetxt(cfg.test_fixed_csv, test_files_fixed, fmt='%s', header='path')

        np.savetxt(cfg.train_moving_csv, train_files_moving, fmt='%s', header='path')
        np.savetxt(cfg.vald_moving_csv, vald_files_moving, fmt='%s', header='path')
        np.savetxt(cfg.test_moving_csv, test_files_moving, fmt='%s', header='path')

        if cfg.seg_available:
            np.savetxt(cfg.train_fixed_seg_csv, train_files_seg_fixed, fmt='%s', header='path')
            np.savetxt(cfg.vald_fixed_seg_csv, vald_files_seg_fixed, fmt='%s', header='path')
            np.savetxt(cfg.test_fixed_seg_csv, test_files_seg_fixed, fmt='%s', header='path')

            np.savetxt(cfg.train_moving_seg_csv, train_files_seg_moving, fmt='%s', header='path')
            np.savetxt(cfg.vald_moving_seg_csv, vald_files_seg_moving, fmt='%s', header='path')
            np.savetxt(cfg.test_moving_seg_csv, test_files_seg_moving, fmt='%s', header='path')

        cfg.num_train_files = train_indices.size

        print(str(train_indices.size) + ' train cases, '
                + str(test_indices.size)
                + ' test cases, ' + str(vald_indices.size) + ' vald cases')

        steps_per_epoch = cfg.num_train_files // batch_size
        distutils.dir_util.mkpath(logs_path+'/'+ str(f))

        if is_training:
            training(f, architecture, variant, losses, loss_weights, learning_rate, nb_features, nb_epochs,
                     batch_size, seed=f)

        if is_pretrained:
            pretrained(f, architecture, variant, losses, loss_weights, learning_rate, nb_features, nb_epochs,
                       steps_per_epoch, batch_size, seed=f)

        if is_apply:
            apply(f, architecture, variant, nb_features, batch_size, seed=f)

        if is_evaluate:
            evaluate(f)

    if is_evaluate:
        evaluation.combine_evaluation_results_from_folds(logs_path+'eval/')
        evaluation.combine_evaluation_results_in_file(logs_path+'eval/')
        evaluation.make_boxplot_graphic(logs_path+'eval/')

#main

is_training = True
is_pretrained = False
is_apply = True
is_evaluate= True
is_different_losses=True

data_fixed, data_moving= loadsave.getdatalist_from_csv(cfg.fixed_csv, cfg.moving_csv)

cfg.height=256
cfg.width=256
cfg.numb_slices=64

batch_size=1

nb_epochs = 200

# Proposed multistage network:
# rigid subnetwork
architecture='adapted_Guetal'
variant='rigid'
#rigid and affine subnetworks
architecture='multistage'
variant='ra'
#full network
architecture='multistage'
variant='rad'

# Benchmark network:
# affine subnetwork
architecture='adapted_Guetal'
variant='affine'
#full network
architecture='multistage'
variant='ad'

nb_features=[[[],[512, 256, 128, 64, 32, 6]],                #rigid
            [[],[512, 256, 128, 64, 32, 12]],                #affine
            [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]] #deformable
            ]

#for MI:
bin_centers_mi_glob = np.linspace(0, 1 - 1 / cfg.nb_bins_glob, cfg.nb_bins_glob)
bin_centers_mi_glob = bin_centers_mi_glob + 0.5 * (bin_centers_mi_glob[1] - bin_centers_mi_glob[0])

learning_rate = 1e-4

losses = [vxm.losses.NMI(bin_centers_mi_glob, (cfg.height, cfg.width, cfg.numb_slices)).loss, vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.1]

loss_names="NMI_GradL2"

distutils.dir_util.mkpath(logs_path)

experiment_1(data_fixed, data_moving, architecture, variant, losses, loss_weights, learning_rate, nb_features, nb_epochs,
                 batch_size, is_training=is_training,
                 is_pretrained=is_pretrained, is_apply=is_apply, is_evaluate=is_evaluate)



