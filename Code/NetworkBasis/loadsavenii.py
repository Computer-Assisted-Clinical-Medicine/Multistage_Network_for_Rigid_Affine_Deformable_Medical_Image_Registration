import os
import numpy as np
import processing
import pandas as pd
import distutils.dir_util
from scipy.ndimage import zoom
import SimpleITK as sitk

from NetworkBasis import config as cfg
import NetworkBasis.image as image
from processing import seg_binary

def load_image(path,filename, do_resample=True, do_normalize=True):
    '''
    Loads Nifti images and returns a numpy array.

    @param path: The path to the Nifti file
    @param filename: The name of the Nifti file
    @return: A numpy array containing the pixel data of the Nifti file
    '''
    # Use a SimpleITK reader to load the nii images and labels for training
    data_img = sitk.ReadImage(os.path.join(path, filename), sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(data_img)
    cfg.data_background_value=int(np.min(data))

    #Resample
    if do_resample:
        data_img= resample(data_img)
    data = sitk.GetArrayFromImage(data_img)

    # Determine the data type based on the path
    if path ==cfg.path_fixed:
        data_type="fixed"
    elif path==cfg.path_moving:
        data_type = "moving"
    elif path==cfg.path_seg_fixed or path==cfg.path_seg_moving:
        data_type = "label"
    else:
        data_type = "moved"

    if do_normalize:
        data = normalize(data, data_type=data_type)

    # move z axis to last index
    data = np.moveaxis(data, 0, -1)
    if cfg.print_details:
        print(data.shape)
    return data

def load_image_data_info_fixed_image(path,filename):
    '''
    Loads Nifti images and returns a numpy array.
    Uses information of fixed image to resample the new image

    @param path: The path to the Nifti file
    @param filename: The name of the Nifti file
    @return: A numpy array containing the pixel data of the Nifti file
    '''
    # Use a SimpleITK reader to load the nii images and labels for training
    data_img = sitk.ReadImage(os.path.join(path, filename))

    data = sitk.GetArrayFromImage(data_img)
    cfg.data_background_value=int(data[data.shape[0]//2:data.shape[0]//2+5, :5, data.shape[2]//2:data.shape[2]//2+5].mean())

    orig_img = sitk.ReadImage(cfg.orig_filepath)
    data_info = image.get_data_info(orig_img)

    # Resample
    data_img= resample_data_info_fixed_image(data_img, data_info)

    data = sitk.GetArrayFromImage(data_img)

    # move z axis to last index
    data = np.moveaxis(data, 0, -1)
    return data

def load_displacementfield(path_disp, filename_moving):
    '''!
        Loads and preprocesses the three-dimensional displacement field given the path to the displacement field file and the name of the moving image file.

        @param path_disp <em>string</em>: the path to the displacement field file.
        @param filename_moving <em>string</em>: the name of the moving image file.

        @return the displacement field as a numpy array.
        '''

    # construct file paths
    path_displacementfield1 = path_disp + "displacement_0_" + filename_moving
    path_displacementfield2 = path_disp + "displacement_1_" + filename_moving
    path_displacementfield3 = path_disp + "displacement_2_" + filename_moving

    # read the three components of the displacement field
    displacementfield1 = sitk.ReadImage(path_displacementfield1)
    displacementfield1_np = processing.img_to_nparray(displacementfield1)
    displacementfield1_np = np.expand_dims(displacementfield1_np, axis=3)

    displacementfield2 = sitk.ReadImage(path_displacementfield2)
    displacementfield2_np = processing.img_to_nparray(displacementfield2)
    displacementfield2_np = np.expand_dims(displacementfield2_np, axis=3)

    displacementfield3 = sitk.ReadImage(path_displacementfield3)
    displacementfield3_np = processing.img_to_nparray(displacementfield3)
    displacementfield3_np = np.expand_dims(displacementfield3_np, axis=3)

    # concatenate the three components into a single numpy array
    displacementfield_np_new = np.append(displacementfield1_np, displacementfield2_np, axis=3)
    displacementfield = np.append(displacementfield_np_new, displacementfield3_np, axis=3)

    return displacementfield


def save_image(data,path,filename, use_orig_file_info=False, do_resample=True):
    """
        Saves a 3D image in ITK format (.nii.gz) to disk.

        Args:
            data (numpy.ndarray): A 3D numpy array containing the image data.
            path (str): The directory path where the image should be saved.
            filename (str): The filename of the image (without file extension).
            use_orig_file_info (bool): If True, uses the metadata from the original image file
                                       to save the new image (default False).
            do_resample (bool): If True, resamples the image to the target resolution and size
                                specified in the config file (default True).

        Returns:
            None
        """
    # Set data info based on whether to use original file info or config file info
    if use_orig_file_info:
        orig_img = sitk.ReadImage(cfg.orig_filepath)
        data_info = image.get_data_info(orig_img)
    else:
        data_info = {}
        if cfg.adapt_resolution:
            data_info['target_spacing'] = cfg.target_spacing
            data_info['target_size'] = cfg.target_size
            data_info['target_type_image'] = cfg.target_type_image

    # Set remaining data info fields
    data_info['res_spacing'] = cfg.target_spacing
    data_info['res_origin'] = [0.0, 0.0, 0.0]
    data_info['res_direction'] = cfg.target_direction

    # Move axis of data array to match ITK format
    data = np.moveaxis(data, -1, 0)

    # Convert numpy array to ITK image object and save to disk
    data_out = image.np_array_to_itk_image(data, data_info, do_resample=do_resample,
                                           out_type=sitk.sitkVectorFloat32,
                                           background_value=cfg.label_background_value,
                                           interpolator=sitk.sitkLinear)
    sitk.WriteImage(data_out, os.path.join(path,filename))


def save_disp(data,path,filename, use_orig_file_info=False, do_resample=True, i=0):
    """
        Saves a 3D image in ITK format (.nii.gz) to disk.

        Args:
            data (numpy.ndarray): A 3D numpy array containing the image data.
            path (str): The directory path where the image should be saved.
            filename (str): The filename of the image (without file extension).
            use_orig_file_info (bool): If True, uses the metadata from the original image file
                                       to save the new image (default False).
            do_resample (bool): If True, resamples the image to the target resolution and size
                                specified in the config file (default True).

        Returns:
            None
        """
    # Set data info based on whether to use original file info or config file info
    if use_orig_file_info:
        orig_img = sitk.ReadImage(cfg.orig_filepath)
        data_info = image.get_data_info(orig_img)
    else:
        data_info = {}
        if cfg.adapt_resolution:
            data_info['target_spacing'] = cfg.target_spacing
            data_info['target_size'] = cfg.target_size
            data_info['target_type_image'] = cfg.target_type_image

    # Set remaining data info fields
    data_info['res_spacing'] = cfg.target_spacing
    data_info['res_origin'] = [0.0, 0.0, 0.0]
    data_info['res_direction'] = cfg.target_direction

    # create the zoom factors
    zoom_factor = (data_info['res_spacing'][0]/data_info['orig_spacing'][0],data_info['res_spacing'][1]/data_info['orig_spacing'][1],data_info['res_spacing'][2]/data_info['orig_spacing'][2])
    data = zoom(data, zoom_factor)

    #also change magnitude of displacementfield according to the zoom factor for resizing
    data= data * zoom_factor[i]

    # Move axis of data array to match ITK format
    data = np.moveaxis(data, -1, 0)

    img = sitk.GetImageFromArray(data)
    img = sitk.Cast(img, sitk.sitkVectorFloat32)
    img.SetSpacing(data_info['orig_spacing'])
    img.SetOrigin(data_info['orig_origin'])
    img.SetDirection(data_info['orig_direction'])
    sitk.WriteImage(img, os.path.join(path,filename))

def save_pred_disp(val_pred, logs_path, filename):
    """
        Saves the predicted displacement fields to disk.

        Args:
        - val_pred (list): A list containing the predicted images and displacement fields.
        - logs_path (str): Path to the directory where the images should be saved.
        - filename (str): Name of the file to save the images to.

        Returns:
        - None
        """

    # Extract the displacement fields
    images_displacement = [img[0,:cfg.height, :cfg.width, :cfg.numb_slices, :] for img in val_pred]

    # Create a directory to save the displacement fields
    distutils.dir_util.mkpath(logs_path + "displacementfields/")

    if cfg.print_details:
        print("images_displacement.shape",images_displacement[1].shape)

    # Extract the 3 displacement fields
    displacement_0 = images_displacement[1][:, :, :, 0]
    displacement_1 = images_displacement[1][:, :, :, 1]
    displacement_2 = images_displacement[1][:, :, :, 2]
    cfg.data_background_value=0

    # Save the displacementfields
    save_disp(displacement_0, logs_path + "displacementfields/", "displacement_0_"+filename, use_orig_file_info=True, i=0)
    save_disp(displacement_1, logs_path + "displacementfields/", "displacement_1_" + filename, use_orig_file_info=True, i=1)
    save_disp(displacement_2, logs_path + "displacementfields/", "displacement_2_" + filename, use_orig_file_info=True, i=2)

def resample(data):
    '''!
    This function operates as follows:
    - extract image meta information
    - augmentation is only on in training
    - calls the static function _resample()

    @param data <em>ITK image,  </em> patient image
    @return resampled data and label images
    '''

    target_info = {}
    target_info['target_spacing'] = cfg.target_spacing
    target_info['target_direction'] = cfg.target_direction
    target_info['target_size'] = cfg.target_size
    target_info['target_type_image'] = cfg.target_type_image
    target_info['target_type_label'] = cfg.target_type_image

    do_augment = False
    cfg.max_rotation=0

    data.SetDirection(cfg.target_direction)

    return image.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                     do_adapt_resolution=cfg.adapt_resolution,
                                     do_augment=do_augment,
                                     max_rotation_augment=cfg.max_rotation)

def resample_data_info_fixed_image(data, data_info):
    '''!
    This function operates as follows:
    - extract image meta information
    - augmentation is only on in training
    - calls the static function _resample()

    @param data <em>ITK image,  </em> patient image
    @return resampled data and label images
    '''

    target_info = {}
    target_info['target_spacing'] = data_info['orig_spacing']
    target_info['target_direction'] = data_info['orig_direction']
    target_info['target_size'] = data_info['orig_size']
    #target_info['orig_origin'] = data_info['orig_origin']
    target_info['target_type_image'] = cfg.target_type_image
    target_info['target_type_label'] = cfg.target_type_image

    do_augment = False
    cfg.max_rotation=0

    data.SetDirection(cfg.target_direction)
    data.SetDirection(data_info['orig_direction']) # NIDDK

    img= image.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                     do_adapt_resolution=cfg.adapt_resolution,
                                     do_augment=do_augment,
                                     max_rotation_augment=cfg.max_rotation)

    if np.min(sitk.GetArrayFromImage(img))==np.max(sitk.GetArrayFromImage(img)): #resample didn't work correctly
        data.SetDirection(data_info['orig_direction'])

        img = img.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                          do_adapt_resolution=cfg.adapt_resolution,
                                          do_augment=do_augment,
                                          max_rotation_augment=cfg.max_rotation)

    return img

def normalize(img, eps=np.finfo(np.float).min, data_type="fixed"):
    '''
    Truncates input to interval [config.norm_min_v, config.norm_max_v] an
     normalizes it to interval [-1, 1] when using WINDOW and to mean = 0 and std = 1 when MEAN_STD.
    '''
    if data_type == "label":
        img=seg_binary(img)

    elif cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
        if data_type=="fixed":
            cfg.norm_min_v = cfg.norm_min_v_fixed
            cfg.norm_max_v = cfg.norm_max_v_fixed
        elif data_type=="moving" or data_type =="moved":
            cfg.norm_min_v = cfg.norm_min_v_moving
            cfg.norm_max_v = cfg.norm_max_v_moving

        flags = img < cfg.norm_min_v
        img[flags] = cfg.norm_min_v
        flags = img > cfg.norm_max_v
        img[flags] = cfg.norm_max_v
        img = (img - cfg.norm_min_v) / (cfg.norm_max_v - cfg.norm_min_v + cfg.norm_eps)
        img = img *cfg.intervall_max # interval [0, 1]

    elif cfg.normalizing_method == cfg.NORMALIZING.MEAN_STD:
        img = img - np.mean(img)
        std = np.std(img)
        img = img / (std if std != 0 else eps)

    elif cfg.normalizing_method == cfg.NORMALIZING.PERCENTMEAN:
        percentile_95 = np.percentile(img, 95)
        percentile_05 = np.percentile(img, 5)
        flags = img > percentile_95
        img[flags] = percentile_95
        flags = img < percentile_05
        img[flags] = percentile_05
        # Mean=0, Std=1
        img = img - np.mean(img)
        std = np.std(img)
        img = img / (std if std != 0 else eps)

    elif cfg.normalizing_method == cfg.NORMALIZING.PERCENTWINDOW:
        percentile_upper = np.percentile(img, 99)
        percentile_lower = np.percentile(img, 5)
        flags = img > percentile_upper
        img[flags] = percentile_upper
        flags = img < percentile_lower
        img[flags] = percentile_lower
        # [0,cfg.intervall_max]
        img = (img - percentile_lower) / (percentile_upper - percentile_lower + cfg.norm_eps)
        img = img * cfg.intervall_max

    if cfg.print_details:
        print("img min and max:",img.min(), img.max())

    return img

def getdatalist_from_csv(fixed_csv, moving_csv):
    """
        Reads two csv files containing the paths of fixed and moving images, and returns two lists of paths respectively.

        Args:
            fixed_csv (str): path to the csv file containing the paths of fixed images
            moving_csv (str): path to the csv file containing the paths of moving images

        Returns:
            tuple: a tuple containing two lists of paths of fixed and moving images respectively
        """
    # Read the csv files into dataframes
    data_list_fixed = pd.read_csv(fixed_csv, dtype=object,sep=';').values
    data_list_moving = pd.read_csv(moving_csv, dtype=object,sep=';').values

    data_fixed=[]
    data_moving=[]

    # Extract the paths from the dataframes and store them into separate lists
    for i in range(len(data_list_fixed)):
        data_fixed.append(data_list_fixed[i][0])
        data_moving.append(data_list_moving[i][0])

    return data_fixed, data_moving

