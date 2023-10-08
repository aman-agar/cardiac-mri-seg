from keras.layers import Conv1D,Conv2D, MaxPooling2D,AveragePooling2D,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation,Flatten,Dropout,Dense, Layer
from keras.layers import Concatenate,Add,Multiply,Reshape,Lambda
from keras.models import Model
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras import Input
from keras.layers import UpSampling2D
from tensorflow import expand_dims,squeeze

# model
def channel_attention(input,units):
  channel = input.shape[-1]
  shared_layer_one = Dense(channel//8,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
  
  shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

  max=GlobalMaxPooling2D()(input)
  max=Reshape((1,1,channel))(max)
  max = shared_layer_one(max)
  max = shared_layer_two(max)
  


  avg=GlobalAveragePooling2D()(input)
  avg=Reshape((1,1,channel))(avg)
  avg = shared_layer_one(avg)
  avg = shared_layer_two(avg)

  input=Add()([max,avg])
  input=Activation('relu')(input)

  return input

def unsqueeze(input):
    return expand_dims(input)

def ssqueeze(input):
    return squeeze(input)

def spatial_attention(input,units):
  max=MaxPooling2D((2,2))(input)
  avg=AveragePooling2D((2,2))(input)
  input=Concatenate()([max,avg])
  input=Conv2D(units,(3,3),padding='same')(input)
  input=BatchNormalization()(input)
  input=Activation('relu')(input)
  input=UpSampling2D(size=(2,2))(input)
  return input

def CBAM(input,units):
  chattn=channel_attention(input,units)
  input=Multiply()([chattn,input])
  spattn=spatial_attention(input,units)
  input=Multiply()([spattn,input])
  return input

def conv(units,input):
  x=BatchNormalization()(input)
  x=Activation('relu')(x)
  x=Conv2D(units,(3,3),padding='same')(x)


  x=BatchNormalization()(x)
  x=Activation('relu')(x)
  x=Conv2D(units,(3,3),padding='same')(x)
  
  short=Conv2D(units,(3,3),padding='same')(input)
  short=BatchNormalization()(short)
  x=Add()([short,x])

  return x

def encoder(units,input):
  x=conv(units,input)
  p=MaxPooling2D((2,2))(x)
  return x,p

def decoder(units,input,skips):
  x=UpSampling2D(size=(2,2))(input)
  x=conv(units,x)
  skips=CBAM(skips,units)
  x=Concatenate()([x,skips])

  return x

def conv2(units,input):
  x=Conv2D(units,(3,3),padding='same')(input)

  
  x=BatchNormalization()(x)
  x=Activation('relu')(x)
  x=Conv2D(units,(3,3),padding='same')(x)
  
  short=Conv2D(units,(3,3),padding='same')(input)
  short=BatchNormalization()(short)
  x=Add()([short,x])

  return x


def encoderV2(units,input):
  x=conv2(units,input)
  # p=MaxPooling2D((2,2))(x)
  return x

def Res_Unet(input_shape):
  inputs=Input(input_shape)
  e1=encoderV2(16,inputs)
  e2,p2=encoder(32,e1)
  e3,p3=encoder(64,p2)
  e4,p4=encoder(128,p3)

  bridge=conv(256,p4)
  bridge=MaxPooling2D((2,2))(bridge)

  d1=decoder(128,bridge,p4)
  d2=decoder(64,d1,p3)
  d3=decoder(32,d2,p2)
  d4=decoder(16,d3,e1)

  output=Conv2D(1,(1,1),activation='sigmoid')(d4)

  model=Model(inputs,output,name='Res-U-Net')

  return model

# Using Model Sub-Classing
class ChannelAttention(Layer):
    '''
    Channel Attention Layer to be used in Block Attention
    '''
    def __init__(self, units):
        super().__init__()

        
    def build(self, input):
        self.channel = input.shape[-1]
        self.shared_layer_one = Dense(self.channel//8,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
        self.shared_layer_two = Dense(self.channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

        self.GMP = GlobalMaxPooling2D()
        self.GMPReshape = Reshape((1,1,self.channel))
        self.GAP = GlobalAveragePooling2D()
        self.GAPReshape = Reshape((1,1,self.channel))

        self.add = Add()
        self.activation = Activation('relu')
         

    def call(self, input):
        # Performing Global Max Pooling
        max = self.GMP(input)
        max = self.GMPReshape(max)
        max = self.shared_layer_one(max)
        max = self.shared_layer_two(max)

        # Performing Global Average Pooling
        avg = self.GAP(input)
        avg = self.GAPReshape(avg)
        avg = self.shared_layer_one(avg)
        avg = self.shared_layer_two(avg)

        # Concatenate the results together and apply RELU
        input = self.add([max, avg])
        input = self.activation(input)
        
        return input
        
class SpatialAttention(Layer):
    '''
    Spatial Attention Layer to be used in Block Attention
    '''
    def __init__(self, units):
        super(SpatialAttention, self).__init__()
        self.max = MaxPooling2D((2,2))
        self.avg = AveragePooling2D((2,2))
        self.concat = Concatenate()
        self.conv1 = Conv2D(units, (3,3), padding = 'same')
        self.bn = BatchNormalization()
        self.activation = Activation('relu')
        self.upSample = UpSampling2D(size=(2,2))
    
    def call(self, input):
        max = self.max(input)
        avg = self.avg(input)
        concat = self.concat([max, avg])
        conv = self.conv1(concat)
        conv = self.bn(conv)
        conv = self.activation(conv)
        conv = self.upSample(conv)

        return conv

class BlockAttention(Layer):
    def __init__(self, units):
        super(BlockAttention, self).__init__()
        self.units = units
        self.CA = ChannelAttention(self.units)
        self.multiply = Multiply()
        self.SA = SpatialAttention(self.units)
        self.multiply2 = Multiply()
        
    def call(self, input):
        CAOut = self.CA(input)
        input = self.multiply([CAOut, input])
        SAOut = self.SA(input)
        input = self.multiply([SAOut, input])

        return input
        
class CustomConv(Layer):
    '''
    Custom Convolution layer consisting of Conv2D, BatchNormalization and Activations
    for Encoder
    '''
    def __init__(self, units):
        super(CustomConv, self).__init__()
        self.units = units
        self.bn1 = BatchNormalization()
        self.activation1 = Activation('relu')
        self.conv1 = Conv2D(self.units, (3,3), padding = 'same')

        self.bn2 = BatchNormalization()
        self.activation2 = Activation('relu')
        self.conv2 = Conv2D(self.units, (3,3), padding = 'same')

        self.conv3 = Conv2D(self.units, (3,3), padding = 'same')
        self.bn3 = BatchNormalization()
        self.add = Add()
    
    def call(self, input):
        
        x = self.bn1(input)
        x = self.activation1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.activation2(x)
        x = self.conv2(x)

        short = self.conv3(input)
        short = self.bn3(short)

        x = self.add([short, x])
        
        return x

class CustomConvV2(Layer):
    '''
    Custom Convolution layer 2 consisting of Conv2D, BatchNormalization and Activations
    for EncoderV2
    '''
    def __init__(self, units):
        super(CustomConvV2, self).__init__()
        
        self.conv1 = Conv2D(units, (3,3), padding = 'same')
        self.bn1 = BatchNormalization()
        self.activ1 = Activation('relu')
        self.conv2 = Conv2D(units, (3,3), padding = 'same')

        self.conv3 = Conv2D(units, (3,3), padding = 'same')
        self.bn2 = BatchNormalization()
        self.add = Add()
    
    def call(self, input):
        x = self.conv1(input)

        x = self.bn1(x)
        x = self.activ1(x)
        x = self.conv2(x)

        short = self.conv3(input)
        short = self.bn2(short)
        
        return self.add([short, x])


class EncoderV2(Layer):
    
    def __init__(self, units):
        super(EncoderV2, self).__init__()
        self.units = units
        self.convV2 = CustomConvV2(self.units)

    def call(self, input):
        return self.convV2(input)

class Encoder(Layer):
    
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units
        self.conv = CustomConv(self.units)
        self.maxPool = MaxPooling2D((2,2))
    
    def call(self, input):
        x = self.conv(input)
        p = self.maxPool(x)
        return x,p

class Decoder(Layer):
    
    def __init__(self, units):
        super(Decoder, self).__init__()
        
        self.units = units 
        self.upSampling = UpSampling2D(size=(2,2))
        self.conv = CustomConv(units)
        self.CBAM = BlockAttention(units)
        self.concat = Concatenate()
    
    def call(self, input, skips):
        x = self.upSampling(input)
        x = self.conv(x)
        skips = self.CBAM(skips)
        return self.concat(x, skips)



class ResUNetCBAM(Model):

    def __init__(self, input_shape):
        super(ResUNetCBAM, self).__init__()

        # self.CAUnits = 32
        self.ev2Units = 32
        
        self.encoderV2 = EncoderV2(self.ev2Units)
        self.encoder1 = Encoder(32)
        self.encoder2 = Encoder(64)
        self.encoder3 = Encoder(128)

        self.bridge = CustomConv(256)
        self.maxPool = MaxPooling2D((2,2))

        self.decoder1 = Decoder(128)
        self.decoder2 = Decoder(64)
        self.decoder3 = Decoder(32)
        self.decoder4 = Decoder(16)

        self.outConv = Conv2D(1,(1,1), activation = 'sigmoid')
        # self.channelAttention = self.channel_attention(self.CAUnits)
    
    def __call__(self, inputs, output):

        e1=self.encoderV2(inputs)
        e2,p2 = self.encoder1(e1)
        e3,p3 = self.encoder2(p2)
        e4,p4 = self.encoder3(p3)

        bridge = self.bridge(p4)
        bridge = self.maxPool(bridge)

        d1 = self.decoder1(bridge,p4)
        d2 = self.decoder2(d1,p3)
        d3 = self.decoder3(d2,p2)
        d4 = self.decoder4(d3,e1)

        output = self.outConv(d4)
        return output
       
       
  