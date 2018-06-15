
import numpy as np
#import cv2
import random
import tensorflow as tf
import cv2
#TEST

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
    
def extract_samples(image, gt, patchshape, number):
    
     # Judge whether gt is in grayscale
     patches = []
     labels = []
     corrs = []
     image1s= np.array(image).astype('float')
     #Convert image to 0-1
     for band in range(0,image.shape[2]):
         temp_band = np.array(image[:,:,band]).astype('float')
         band_range = float(np.max(temp_band) - np.min(temp_band))
         temp_band = np.array((temp_band-np.min(temp_band))/band_range).astype('float')
         image1s[:,:,band]=temp_band
     
     if len(gt.shape)<3:
         for label in np.unique(gt):
             ii = np.nonzero(gt == label)
             all_cor = tuple(zip(*ii))
             #if int(len(all_cor)*number)>2:#!!!!!probably this is wrong,samples are wrong
             if label>0:
                 selected_cor = random.sample(range(1,len(all_cor)), int(len(all_cor)*number))
                 for patch_indx in selected_cor:
                     cen_patch = all_cor[patch_indx]
                     corr = cen_patch
                     a = cen_patch[0]-int(patchshape[0]/2)
                     b = cen_patch[0] + int(patchshape[0]/2)
                     c = cen_patch[1]-int(patchshape[0]/2)
                     d = cen_patch[1]+int(patchshape[1]/2)
                     label = gt[cen_patch[0],cen_patch[1]]
                     if (a>0 and b<=gt.shape[0] and c>0 and d<gt.shape[1]):
                         # Slice tuple indexing the region of our proposed patc
                         region = (slice(a, b),slice(c, d))
                         # The actual pixels in that region.
                         patch = image1s[region]
                         patches.append(patch)
                         labels.append(label)
                         corrs.append(corr)
                     
             else:
                 print("short of training samples!")
     else:
         print("Reference map is not right!")
        
     return (np.concatenate([pat[np.newaxis, ...] for pat in patches], axis=0),labels,corrs)
     
     
def Pre_patches(image, patchshape, resolution):
    bd_size = np.repeat(np.array(np.int(patchshape[0]/2)),4)
    #img = cv2.imread(image_path)
    patches = []
    coors_ys = []
    coors_xs = []
    if resolution<=1:
        if resolution==1:
            shifts = 1
        else:
            shifts = int(patchshape[0]*(1-resolution))
        image = cv2.copyMakeBorder(image,bd_size[0],bd_size[1],
                                 bd_size[2],bd_size[3],cv2.BORDER_REPLICATE)
        rowstart = np.int(patchshape[0]/2)+1
        row_count = 1
        while rowstart<image.shape[0]-np.int(patchshape[0]/2):
            colstart = np.int(patchshape[1]/2)+1
            col_count = 1
            while colstart<image.shape[1]-np.int(patchshape[1]/2):
                region = (slice(rowstart-int(patchshape[0]/2), rowstart + int(patchshape[1]/2)),
                          slice(colstart-int(patchshape[0]/2), colstart + int(patchshape[1]/2)))
                patch = image[region]
                patches.append(patch)
                colstart += shifts
                coors_ys.append(col_count)
                coors_xs.append(row_count)
                col_count +=1
            rowstart += shifts
            row_count+=1
    else: 
        print "No support enlarged image!"
        patch = []
    
    return (np.concatenate([pat[np.newaxis, ...] for pat in patches], axis=0), coors_xs, coors_ys)

def frac_eq_to(image, value=0):
    return (image == value).sum() / float(np.prod(image.shape))



