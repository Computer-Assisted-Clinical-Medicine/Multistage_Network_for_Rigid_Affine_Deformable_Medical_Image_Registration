import tensorflow.keras.layers as KL
from tensorflow.keras import layers
import voxelmorph as vxm
from voxelmorph import layers
# TODO: change full module imports as opposed to specific function imports
from voxelmorph.tf.modelio import LoadableModel, store_config_args

from NetworkBasis.layers import *

# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

class Model_adapted_Guetal(LoadableModel):
    """
    Two-Stage Unsupervised Learning Method for Affine and Deformable Medical Image Registration
    """

    @store_config_args
    def __init__(self,inshape, name=""):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """
        # configure unet input shape (concatenation of moving and fixed images)
        ndim = 3
        unet_input_features = 2
        src_feats = 1
        trg_feats = 1

        nb_features= [8, 16, 32, 64, 128]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        x=net_input
        for i in range(len(nb_features)):
            x= KL.Conv3D(nb_features[i],kernel_size=3, padding="SAME", name="conv"+str(i))(x)
            x= KL.MaxPooling3D(2, name="conv"+str(i)+"_pooling")(x)
            x = KL.LeakyReLU(0.2, name="conv" + str(i) + "_activation")(x)

        x = KL.Flatten()(x)

        if name=="_rigid":
            parameters_aff = KL.Dense(6)(x)
            parameters_aff = vxm.layers.AffineTransformationsToMatrix(ndims=3,name='rigidparameterstomatrix')(parameters_aff)
        elif name =="_affine":
            parameters_aff = KL.Dense(12)(x)

        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Guetal'+name, inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        #self.references.net_model = net
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_multistage(LoadableModel):
    """
    Neural Network for (unsupervised) registration between two images.
    combine rigid, affine and deformable transformation
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_features, variant):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: encoder and decoder.
        """
        # configure unet input shape (concatenation of moving and fixed images)
        src_feats = 1
        trg_feats = 1

        if variant=="ra":
            use_rigid = True
            use_affine = True
            use_deformable = False
        elif variant == "rad":
            use_rigid = True
            use_affine  = True
            use_deformable = True
        elif variant == "ad":
            use_rigid = False
            use_affine = True
            use_deformable = True
            nb_features=nb_features[1:]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        moved_image=source

        if use_rigid:  # rigid transformation
            output = Model_adapted_Guetal(moved_image.shape[1:-1], name="_rigid")([moved_image, target])
            disp_sum = output[1]
            moved_image = output[0]


        if use_affine:  # affine transformation
            output = Model_adapted_Guetal(moved_image.shape[1:-1], name="_affine")([moved_image, target])

            if use_rigid == False:
                disp_sum = output[1]
            else:
                disp_sum = disp_sum + output[1]
            moved_image = vxm.layers.SpatialTransformer(name='transformer_affine')([source, disp_sum])

        if use_deformable:  # deformable transformation
            output = vxm.networks.VxmDense(inshape=moved_image.shape[1:-1], nb_unet_features=nb_features[-1], int_downsize=1)([moved_image, target])
            disp_sum = disp_sum + output[1]
            moved_image = vxm.layers.SpatialTransformer(name='transformer_deformable')([source, disp_sum])

        outputs = [moved_image, disp_sum]

        super().__init__(name='multistage', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        # self.references.net_model = net
        self.references.y_source = moved_image
        self.references.pos_flow = disp_sum

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])