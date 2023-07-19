import tensorflow as tf
import os
import voxelmorph as vxm
import numpy as np
import distutils.dir_util
import skimage.transform
import SimpleITK as sitk
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import shift
from NetworkBasis import config as cfg
import NetworkBasis.loadsavenii as loadsave

def pad_img(image, ref_image):
    """
    Pads the input `image` array with zeros to match the shape of the `ref_image` array along each dimension.

    Args:
        image (numpy.ndarray): The input image array to pad.
        ref_image (numpy.ndarray): The reference image array whose shape the input `image` array will be padded to match.

    Returns:
        tuple: A tuple consisting of the padded image array with the same shape as `ref_image`, and the amount of padding added along each dimension.
    """

    # Calculate the amount of padding to add to each dimension of the image array
    pad_y = ref_image.shape[0] - image.shape[0]
    pad_x = ref_image.shape[1] - image.shape[1]
    pad_z = ref_image.shape[2] - image.shape[2]

    # If any of the padding amounts are negative, set them to zero (no padding needed along that dimension)
    if pad_x < 0:
        pad_x = 0
    if pad_y < 0:
        pad_y = 0
    if pad_z < 0:
        pad_z = 0

    # Create a tuple of tuples representing the amount of padding to add to each dimension of the image array
    pad_amount = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2), (pad_z // 2, pad_z - pad_z // 2))

    # Pad the image array with zeros using the `pad_amount` tuple and the `constant` mode
    image = np.pad(image, pad_amount, 'constant', constant_values=int(image[image.shape[0]//2:image.shape[0]//2+5, :5, image.shape[2]//2:image.shape[2]//2+5].mean()))

    # Return a tuple consisting of the padded image array and the amount of padding added along each dimension
    return image, pad_y, pad_x, pad_z

def img_to_nparray(img):
    """
        Converts a SimpleITK image object to a numpy array.

        Args:
            img (SimpleITK image object): The image.

        Returns:
            image data (numpy array).
        """
    data = sitk.GetArrayFromImage(img)
    data = np.moveaxis(data, 0, -1)
    return data

def nparray_to_img(img):
    """
        Converts a numpy array to a SimpleITK image object.

        Args:
            img (numpy array).

        Returns:
            image data (SimpleITK image object).
        """

    img = np.moveaxis(img, -1, 0)
    img = sitk.GetImageFromArray(img)
    return img

def warp_img(path_result, filename_fixed, filename_moving, path_moving=cfg.path_moving):
    """
        Warps a moving image using a displacement field. Moved image is resized to fixed image shape .

        Args:
            displacementfield: A displacement field numpy array.
            path_result: A string representing the path to save the warped image.
            filename_fixed: A string representing the filename of the fixed image.
            filename_moving: A string representing the filename of the moving image.
            path_moving: A string representing the path of the moving image. Default is set in the cfg file.

        Returns:
            None.
        """

    # Load the moving image and displacementfield.
    fixed_np = loadsave.load_image(cfg.path_fixed, filename_fixed, do_resample=False ,do_normalize=False)
    moving_np = loadsave.load_image_data_info_fixed_image(path_moving, filename_moving)
    displacementfield= loadsave.load_displacementfield(path_result+"/displacementfields/",filename_moving)

    if cfg.print_details:
        print(" warp_img_fixed_size",moving_np.shape)
        print(" warp_img_fixed_size displacementfield", displacementfield.shape)


    #pad moving image
    fixed_shape_orig=fixed_np.shape

    moving_np, pad_y, pad_x, pad_z = pad_img(moving_np, displacementfield)

    fixed_np, _, _, _ = pad_img(fixed_np, displacementfield)

    displacementfield = np.expand_dims(displacementfield, axis=0)

    moving_np = np.expand_dims(moving_np, axis=0)
    moving_np = np.expand_dims(moving_np, axis=4)

    # Build the transformer layer.
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

    # Warp the moving image using the displacement field.
    moved_np = spatial_transformer(
        [tf.convert_to_tensor(moving_np), tf.convert_to_tensor(displacementfield)])

    if cfg.print_details:
        print("image_warped shape", moved_np.shape)

    moved_np = moved_np[0, :, :, :, 0]

    moved_np = moved_np[pad_y // 2:pad_y // 2 + fixed_shape_orig[0], pad_x // 2:pad_x // 2 + fixed_shape_orig[1],
        pad_z // 2:pad_z // 2 + fixed_shape_orig[2]]

    if moved_np.shape[2] < fixed_shape_orig[2]:
        pad_z= fixed_shape_orig[2] - moved_np.shape[2]
        pad_amount = ((0, 0), (0, 0), (pad_z // 2, pad_z - pad_z // 2))

        # Pad the image array with zeros using the `pad_amount` tuple and the `constant` mode
        moved_np = np.pad(moved_np, pad_amount, 'constant', constant_values=int(
            tf.reduce_mean(moved_np[moved_np.shape[0] // 2:moved_np.shape[0] // 2 + 5, :5, moved_np.shape[2] // 2:moved_np.shape[2] // 2 + 5])))
        print("moved image padded")

    loadsave.save_image(moved_np, path_result, "moved_"+filename_moving, use_orig_file_info=True, do_resample=False)

def seg_binary(img_np,threshold=0.5):
    """
        Converts an input image array to binary, with a given threshold.

        Args:
        - img_np (numpy array): input image array, with shape (height, width, channels)
        - threshold (float): threshold value for segmentation (default is 0.5)

        Returns:
        - img_bin (numpy array): binary image array, with shape (height, width, channels)
        """
    img_bin = np.where(img_np < threshold, 0, 1)
    return img_bin

def warp_seg(path_result,filename_fixed, filename_seg_moving, path_seg_moving=cfg.path_seg_moving, filename_moving=" "):
    """
        Applies a warp to a segmentation image using a displacement field and resizes the output to the size of the
        fixed image.

        Args:
        - displacementfield (numpy array): the displacement field for the warp
        - predict_path (string): path to save the warped segmentation
        - filename_seg_fixed (string): the filename of the fixed segmentation
        - filename_seg_moving (string): the filename of the moving segmentation
        - path_seg_moving (string): the path of the moving segmentation. Default: cfg.path_seg_moving
        - filename_moving (string): the filename of the moving image. Default: " "

        Returns:
        - None
        """
    distutils.dir_util.mkpath(path_result)

    # Load the moving and segmentation image and moving and fixed image and displacementfield
    fixed_np = loadsave.load_image(cfg.path_fixed, filename_fixed, do_resample=False, do_normalize=False)
    moving_np_seg = loadsave.load_image_data_info_fixed_image(path_seg_moving, filename_seg_moving)

    displacementfield = loadsave.load_displacementfield(path_result + "../displacementfields/", filename_moving)

    # pad moving image
    fixed_shape_orig = fixed_np.shape

    moving_np_seg, pad_y, pad_x, pad_z = pad_img(moving_np_seg, displacementfield)
    fixed_np, _, _, _ = pad_img(fixed_np, displacementfield)

    displacementfield = np.expand_dims(displacementfield, axis=0)

    moving_np_seg = np.expand_dims(moving_np_seg, axis=0)
    moving_np_seg = np.expand_dims(moving_np_seg, axis=4)

    # Build the transformer layer
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

    # Warp the moving segmentation using the displacement field
    seg_warped = spatial_transformer([tf.convert_to_tensor(moving_np_seg, dtype=tf.float32),
                                      tf.convert_to_tensor(displacementfield, dtype=tf.float32)])

    seg_warped = seg_warped[0, :, :, :, 0]

    # crop image

    seg_warped = seg_warped[pad_y // 2:pad_y // 2 + fixed_shape_orig[0], pad_x // 2:pad_x // 2 + fixed_shape_orig[1],
            pad_z // 2:pad_z // 2 + fixed_shape_orig[2]]
    seg_warped = seg_binary(seg_warped)
    loadsave.save_image(seg_warped, path_result, "seg_"+filename_seg_moving, use_orig_file_info=True, do_resample=False)

