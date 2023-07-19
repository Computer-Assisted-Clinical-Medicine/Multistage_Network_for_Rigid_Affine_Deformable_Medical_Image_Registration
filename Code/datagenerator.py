import numpy as np
import tensorflow

from NetworkBasis import config as cfg
import NetworkBasis.loadsavenii as loadsave

class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, fixed_filenames, path_fixed, moving_filenames, path_moving, batch_size,shape, model='vxm', variant=" "):
        self.fixed_filenames = fixed_filenames
        self.moving_filenames= moving_filenames
        self.path_fixed = path_fixed
        self.path_moving = path_moving
        self.batch_size = batch_size
        self.shape=shape
        self.model=model
        self.variant=variant

    def __len__(self):
        return (np.ceil(len(self.fixed_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_fixed = self.fixed_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_moving = self.moving_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        vol_shape = self.shape
        ndims = len(self.shape)
        zero_phi = np.zeros([self.batch_size, *vol_shape, ndims])

        fixed_images = np.array([loadsave.load_image(self.path_fixed, file_name[0][2:-1]) for file_name in batch_fixed])
        if cfg.print_details:
            print(batch_fixed[0][0][2:-1], batch_moving[0][0][2:-1])
        moving_images = np.array([loadsave.load_image(self.path_moving, file_name[0][2:-1]) for file_name in batch_moving])

        fixed_images = fixed_images[0, :, :, :]
        moving_images = moving_images[0, :, :, :]

        fixed_images = np.expand_dims(fixed_images, axis=0)
        moving_images = np.expand_dims(moving_images, axis=0)

        moving_images = np.expand_dims(moving_images, axis=4)
        fixed_images = np.expand_dims(fixed_images, axis=4)
        inputs = [moving_images, fixed_images]

        outputs = [fixed_images, zero_phi]

        return (inputs, outputs)

def get_test_images(fixed_filename, path_fixed, moving_filename, path_moving, batch_size,shape, model='vxm'):

    fixed_image = loadsave.load_image(path_fixed, fixed_filename[2:-1])
    moving_image = loadsave.load_image(path_moving, moving_filename[2:-1])

    fixed_image = np.expand_dims(fixed_image, axis=0)
    moving_image = np.expand_dims(moving_image, axis=0)
    moving_image = np.expand_dims(moving_image, axis=4)
    fixed_image = np.expand_dims(fixed_image, axis=4)
    inputs = [moving_image, fixed_image]

    return (inputs)
