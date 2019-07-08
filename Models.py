import numpy as np
from keras.layers import Input, Conv2D, concatenate, add, Lambda, Cropping2D
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D, Conv3D, Reshape, MaxPooling2D
from keras.layers import Flatten, Dense, ZeroPadding3D
from keras.layers import GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.models import Model
from keras.initializers import RandomNormal

# Parameterized 2D Block Model
def BlockModel2D(input_shape,filt_num=16,numBlocks=3):
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
    
    # check for input shape compatibility
    rows,cols = input_shape[0:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"
    
    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    assert minsize>4, "Too small of input for this many blocks. Use fewer blocks or larger input"
    
    # input layer
    lay_input = Input(shape=input_shape,name='input_layer')
    
    # contracting blocks
    x = lay_input
    skip_list = []
    for rr in range(1,numBlocks+1):
        x1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(x)
        x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(x)
        x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(x)
        x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(x51)
        x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1,x3,x52],name='merge_{}'.format(rr))
        x = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = ZeroPadding2D(padding=(1,1),name='PrePad_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='DownSample_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (3, 3),padding='same',name='ConvClean_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_clean_{}'.format(rr))(x)
        skip_list.append(x)
        
        
    # expanding blocks
    expnums = list(range(1,numBlocks+1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd-1],x],name='skip_connect_{}'.format(dd))
        x1 = Conv2D(filt_num*dd, (1, 1),padding='same',name='DeConv1_{}'.format(dd))(x)
        x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_Dx1_{}'.format(dd))(x1)
        x3 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv3_{}'.format(dd))(x)
        x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_Dx3_{}'.format(dd))(x3)
        x51 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(x)
        x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_Dx51_{}'.format(dd))(x51)
        x52 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(x51)
        x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_Dx52_{}'.format(dd))(x52)
        x = concatenate([x1,x3,x52],name='Dmerge_{}'.format(dd))
        x = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_Dall_{}'.format(dd))(x)
        x = UpSampling2D(size=(2,2),name='UpSample_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConvClean1_{}'.format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean1_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConvClean2_{}'.format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean2_{}'.format(dd))(x)
        
    # classifier
    lay_out = Conv2D(output_chan,(1,1), activation='sigmoid',name='output_layer')(x)
    
    return Model(lay_input,lay_out)