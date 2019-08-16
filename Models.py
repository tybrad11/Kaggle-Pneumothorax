import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.initializers import RandomNormal
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Cropping2D, Dense, Flatten, GlobalAveragePooling2D,
                          Input, Lambda, MaxPooling2D, Reshape, UpSampling2D,
                          ZeroPadding2D, ZeroPadding3D, Add, concatenate)
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.models import Model


# Parameterized 2D Block Model
def BlockModel2D(input_shape, filt_num=16, numBlocks=3):
    """Creates a Block CED model for segmentation problems
    Args:
        input shape: a list or tuple of [rows,cols,channels] of input images
        filt_num: the number of filters in the first and last layers
        This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
        regression: Whether to have a continuous output with linear activation
    Returns:
        An unintialized Keras model

    Example useage: SegModel = BlockModel2D([256,256,1],filt_num=8)

    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
    the rows/cols must be divisible by 2^numBlocks for skip connections
    to match up properly
    """
    use_bn = True

    # check for input shape compatibility
    rows, cols = input_shape[0:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize-np.sum(2**np.arange(1, numBlocks+1)))/2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name='input_layer')

    # contracting blocks
    x = lay_input
    skip_list = []
    for rr in range(1, numBlocks+1):
        x1 = Conv2D(filt_num*rr, (1, 1), padding='same',
                    name='Conv1_{}'.format(rr))(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*rr, (3, 3), padding='same',
                    name='Conv3_{}'.format(rr))(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv51_{}'.format(rr))(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv52_{}'.format(rr))(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1, x3, x52], name='merge_{}'.format(rr))
        x = Conv2D(filt_num*rr, (1, 1), padding='valid',
                   name='ConvAll_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name='PrePad_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (4, 4), padding='valid',
                   strides=(2, 2), name='DownSample_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (3, 3), padding='same',
                   name='ConvClean_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_clean_{}'.format(rr))(x)
        skip_list.append(x)

    # expanding blocks
    expnums = list(range(1, numBlocks+1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd-1], x],
                            name='skip_connect_{}'.format(dd))
        x1 = Conv2D(filt_num*dd, (1, 1), padding='same',
                    name='DeConv1_{}'.format(dd))(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_Dx1_{}'.format(dd))(x1)
        x3 = Conv2D(filt_num*dd, (3, 3), padding='same',
                    name='DeConv3_{}'.format(dd))(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_Dx3_{}'.format(dd))(x3)
        x51 = Conv2D(filt_num*dd, (3, 3), padding='same',
                     name='DeConv51_{}'.format(dd))(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_Dx51_{}'.format(dd))(x51)
        x52 = Conv2D(filt_num*dd, (3, 3), padding='same',
                     name='DeConv52_{}'.format(dd))(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_Dx52_{}'.format(dd))(x52)
        x = concatenate([x1, x3, x52], name='Dmerge_{}'.format(dd))
        x = Conv2D(filt_num*dd, (1, 1), padding='valid',
                   name='DeConvAll_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dall_{}'.format(dd))(x)
        x = UpSampling2D(size=(2, 2), name='UpSample_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3), padding='same',
                   name='DeConvClean1_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean1_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3), padding='same',
                   name='DeConvClean2_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean2_{}'.format(dd))(x)

    # classifier
    lay_out = Conv2D(1, (1, 1), activation='sigmoid', name='output_layer')(x)

    return Model(lay_input, lay_out)

    # Parameterized 2D Block Model


def BlockModel_Classifier(input_shape, filt_num=16, numBlocks=3):
    """Creates a Block model for pretraining on classification task
    Args:
        input shape: a list or tuple of [rows,cols,channels] of input images
        filt_num: the number of filters in the first and last layers
        This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
        regression: Whether to have a continuous output with linear activation
    Returns:
        An unintialized Keras model

    Example useage: SegModel = BlockModel2D([256,256,1],filt_num=8)

    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
    the rows/cols must be divisible by 2^numBlocks for skip connections
    to match up properly
    """

    use_bn = True

    # check for input shape compatibility
    rows, cols = input_shape[0:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize-np.sum(2**np.arange(1, numBlocks+1)))/2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name='input_layer')

    # contracting blocks
    x = lay_input
    skip_list = []
    for rr in range(1, numBlocks+1):
        x1 = Conv2D(filt_num*rr, (1, 1), padding='same',
                    name='Conv1_{}'.format(rr))(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*rr, (3, 3), padding='same',
                    name='Conv3_{}'.format(rr))(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv51_{}'.format(rr))(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv52_{}'.format(rr))(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1, x3, x52], name='merge_{}'.format(rr))
        x = Conv2D(filt_num*rr, (1, 1), padding='valid',
                   name='ConvAll_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name='PrePad_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (4, 4), padding='valid',
                   strides=(2, 2), name='DownSample_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (3, 3), padding='same',
                   name='ConvClean_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_skip_{}'.format(rr))(x)

    # average pooling
    x = GlobalAveragePooling2D()(x)
    # classifier
    lay_out = Dense(1, activation='sigmoid', name='output_layer')(x)

    return Model(lay_input, lay_out)


def ConvertEncoderToCED(model):
    # Returns a model with frozen encoder layers
    # and complimentary, unfrozen decoder layers
    # get input layer
    # model must be compiled again after using this function
    lay_input = model.input
    # get skip connection layer outputs
    skip_list = [l.output for l in model.layers if 'skip' in l.name]
    numBlocks = len(skip_list)
    filt_num = int(skip_list[0].shape[-1])
    x = model.layers[-3].output
    # freeze encoder layers
    for layer in model.layers:
        layer.trainable = False
    
    use_bn = True

    # make expanding blocks
    expnums = list(range(1, numBlocks+1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd-1], x],
                            name='skip_connect_{}'.format(dd))
        x1 = Conv2D(filt_num*dd, (1, 1), padding='same',
                    name='DeConv1_{}'.format(dd))(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_Dx1_{}'.format(dd))(x1)
        x3 = Conv2D(filt_num*dd, (3, 3), padding='same',
                    name='DeConv3_{}'.format(dd))(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_Dx3_{}'.format(dd))(x3)
        x51 = Conv2D(filt_num*dd, (3, 3), padding='same',
                     name='DeConv51_{}'.format(dd))(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_Dx51_{}'.format(dd))(x51)
        x52 = Conv2D(filt_num*dd, (3, 3), padding='same',
                     name='DeConv52_{}'.format(dd))(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_Dx52_{}'.format(dd))(x52)
        x = concatenate([x1, x3, x52], name='Dmerge_{}'.format(dd))
        x = Conv2D(filt_num*dd, (1, 1), padding='valid',
                   name='DeConvAll_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dall_{}'.format(dd))(x)
        x = UpSampling2D(size=(2, 2), name='UpSample_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3), padding='same',
                   name='DeConvClean1_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean1_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3), padding='same',
                   name='DeConvClean2_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean2_{}'.format(dd))(x)

    # classifier
    lay_out = Conv2D(1, (1, 1), activation='sigmoid', name='output_layer')(x)

    return Model(lay_input, lay_out)


def Inception_model(input_shape=(299, 299, 3)):
    incep_model = InceptionV3(
        include_top=False, weights=None, input_shape=input_shape, pooling='avg')
    input_layer = incep_model.input
    incep_output = incep_model.output
    # x = Conv2D(16, (3, 3), activation='relu')(incep_output)
    # x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(incep_output)
    return Model(inputs=input_layer, outputs=x)

def res_unet(input_shape):
    
    input_tensor = Input(shape=(input_shape),name='input_layer')
    x = Conv2D(24, (3,3), activation='relu', padding='same')(input_tensor)
    res1 = Conv2D(24, (1,1), activation='relu', padding='same')(input_tensor) #to be added at end of block1
    x = BatchNormalization()(x)
    x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    block1 = Add()([x,res1]) #to be contatenated with decoder
    
    x = MaxPooling2D((2, 2))(block1)
    res2 = Conv2D(48, (1,1), activation='relu', padding='same')(x)#to be added at end of block2
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    block2 = Add()([x, res2])#to be contatenated with decoder
    
    x = MaxPooling2D((2, 2))(block2)
    res3 = Conv2D(96, (1,1), activation='relu', padding='same')(x)#to be added at end of block3
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    block3 = Add()([x, res3])#to be contatenated with decoder
    
    x = MaxPooling2D((2, 2))(block3)
    res4 = Conv2D(192, (1,1), activation='relu', padding='same')(x)#to be added at end of block4
    x = Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, res4])# now go to Decoder side -> UpSampling
    
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, block3])
    res5 = Conv2D(96, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,res5])
    
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, block2])
    res6 = Conv2D(48, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,res6])
    
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, block1])
    res7 = Conv2D(24, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
    x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,res7])
    
    # end part
    x = Conv2D(1, (1,1), activation='relu', padding='same')(x)
    res_input = Conv2D(1, (1,1), activation='relu', padding='same')(input_tensor)#this seems likely the step, although the graph isn’t clear on this
    x = Add()([x, res_input])
    
    out = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)
    
    return Model(inputs=input_tensor, outputs=out)

