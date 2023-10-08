import os
import nibabel as nib
import cv2
import numpy as np


path=r"ACDC\training"


def nii_to_image(nii_file_path,filename):
    img=nib.load(nii_file_path)
    img_fdata=img.get_fdata()

    img_f_path=r"Heart Segmentation Project\AC\Images"

    if not os.path.exists(img_f_path):
        os.mkdir(img_f_path)
    (x,y,z)=img.shape
    for i in range(z):
        slice=img_fdata[:,:,i]
        cv2.imwrite(os.path.join(img_f_path,filename+'slice_{}.png'.format(i)),slice)

def nii_to_mask(nii_file_path,filename):
    img=nib.load(nii_file_path)
    img_fdata=img.get_fdata()
    img_f_path=r"Heart Segmentation Project\AC\Mask"

    if not os.path.exists(img_f_path):
        os.mkdir(img_f_path)
    (x,y,z)=img.shape
    for i in range(z):
        slice=img_fdata[:,:,i]
        slice=np.where(slice==1,85,slice)
        slice=np.where(slice==2,170,slice)
        slice=np.where(slice==3,255,slice)
        cv2.imwrite(os.path.join(img_f_path,filename+'slice_{}.png'.format(i)),slice)


def mask_dir_gen(mask_file_path,filename):
    nii_to_mask(mask_file_path,filename)

def image_dir_gen(img_file_path,filename):
    nii_to_image(img_file_path,filename)


for patient in os.listdir(path):
    if patient=="processed_acdc_dataset":
        continue
    print(patient)
    file_list=[]

    individual_patient_dir = os.path.join(path,patient)
    for file in os.listdir(individual_patient_dir):
        file_list.append(file)

    #file_list=os.listdir(individual_patient_dir)
    file_list.sort()
    img_file_path=os.path.join(individual_patient_dir,file_list[2])
    mask_file_path=os.path.join(individual_patient_dir,file_list[3])
    image_dir_gen(img_file_path,file_list[4])
    mask_dir_gen(mask_file_path,file_list[5])        


    

# def nii_to_image(niifile):
#  filenames = os.listdir(filepath) # Read nii Folder 
#  slice_trans = []
 
#  for f in filenames:
#   # Start reading nii Documents 
#   img_path = os.path.join(filepath, f)
#   img = nib.load(img_path)    # Read nii
#   img_fdata = img.get_fdata()
#   fname = f.replace('.nii','')   # Remove nii Suffix name of 
#   img_f_path = os.path.join(imgfile, fname)
#   # Create nii The folder of the corresponding image 
#   if not os.path.exists(img_f_path):
#    os.mkdir(img_f_path)    # New Folder 
 
#   # Start converting to an image 
#   (x,y,z) = img.shape
#   for i in range(z):      #z Is a sequence of images 
#    silce = img_fdata[i, :, :]   # You can choose which direction of slice 
#    imageio.imwrite(os.path.join(img_f_path,'{}.png'.format(i)), silce)
#             # Save an image 
     

