import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Input
import tensorflow as tf
from keras.models import Model
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
from model import ResUNetCBAM


# Define Generator function 
def generator(img_dir,msk_dir):
    imggen = ImageDataGenerator()
    mskgen = ImageDataGenerator()

    img_gen = imggen.flow_from_directory(directory=img_dir,class_mode=None,shuffle=False,color_mode='grayscale',seed=10,target_size=(224,224))
    msk_gen = mskgen.flow_from_directory(directory=msk_dir,class_mode=None,shuffle=False,color_mode='grayscale',seed=10,target_size=(224,224))

    gen = zip(img_gen,msk_gen)

    for (img,msk) in gen:
        yield (img,msk)

train_img_dir = r'train/Image'
train_msk_dir = r'train/Mask'

val_img_dir = r'val/Image'
val_msk_dir = r'val/Mask'

train_gen = generator(train_img_dir,train_msk_dir)
val_gen = generator(val_img_dir,val_msk_dir)

x_train,y_train = train_gen.__next__()
x_test,y_test = val_gen.__next__()

input_shape = (224,224,1)
inputs = Input(input_shape)
output = (224,224,1)

# Create Model Object
model=Model(inputs,output,name='Res-U-Net')
res_unet=ResUNetCBAM((224,224,1))
res_unet.summary()

# Define Loss Function
def dice_coef2(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    #y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    sum = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (sum + smooth), axis=0)

def dice_coef2_loss(y_true, y_pred, smooth=1.0):
    return 1-dice_coef2(y_true, y_pred, smooth)

# Define optimizer and compile
opt=tf.keras.optimizers.Adam(learning_rate=1e-3)
res_unet.compile(loss=dice_coef2_loss,optimizer=opt,metrics=[dice_coef2])

# Define training arguments
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/Final_Res-U-Net_Channel_on_ACDC.h5',
    save_weights_only=False,
    monitor='val_dice_coef2',
    mode='max',
    save_freq='epoch',
    save_best_only=True)

# Train the model
history=res_unet.fit(train_gen,epochs=47,validation_data=val_gen,steps_per_epoch=32,validation_steps=11.90,callbacks=[model_checkpoint_callback])

# Save the trained model
res_unet.save("Final_Res-U-Net_with_CBAM_attention_on_ACDC.h5")