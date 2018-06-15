import os
import tensorflow as tf
import numpy as np
import sample_extraction as samples # sample_extraction.py
#import cv2
from scipy import misc
import CNN_models as CNNs
#import gdal
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pickle
from numpy import sqrt
from smooth_filter import gaussian_grid, filter_image # include smooth_filter.py
from graph import build_graph, segment_graph # include graph.py
from collections import Counter


###############################################
#Parameter setting

patchshape = [28,28]
batch_size = 500
test_batch_size =1
test_size = 100
max_steps = 30
image_path = os.path.join(os.getcwd(),'ori_3_shrink.tif')
img = misc.imread(image_path)
resolution = 0.8 # Set skip steps during prediction steps = patchsize*(1-resolution)

#Parameter setting for segmentation (graph-based)
sigma = 0.2
neighbor = 4
K = 0.3
min_size = 3 #Set the initial segmentation scale

###############################################


class FLAGS(object):
    def __init__(self):
        self.train = None
        self.test = None
        self.checkpoint_dir = None

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
    
def read_tif_as_array (path):
    img = cv2.imread(path)

    return img
    
def diff_rgb(img, x1, y1, x2, y2):
    r = (img[0][x1, y1] - img[0][x2, y2]) ** 2
    g = (img[1][x1, y1] - img[1][x2, y2]) ** 2
    b = (img[2][x1, y1] - img[2][x2, y2]) ** 2
    return sqrt(r + g + b)

def diff_grey(img, x1, y1, x2, y2):
    v = (img[x1, y1] - img[x2, y2]) ** 2
    return sqrt(v)

def threshold(size, const):
    return (const / size)
    
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def generate_image(forest, width, height):
    #random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    #colors = [random_color() for i in xrange(width*height)]
    #colors_1 = range(width*height)
    img_seg = np.zeros((width, height),dtype=np.int)
    for y in xrange(height):
        for x in xrange(width):
            comp = forest.find(y * width + x)
            img_seg[x][y] = np.int(comp)
        
    return img_seg#.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    
def ocnn(img_seg, width, height, pre_large_im):
    # Convert the segment indices from 1...n
    img_statics = np.zeros((width, height),dtype=np.int)
    temp_unique = np.unique(img_seg)
    for p in xrange(len(temp_unique)):
        temp_value = temp_unique[p]
        idx = np.where(img_seg==temp_value)
        temp_gt = pre_large_im[idx[0],idx[1]]
        temp_label = np.int(Most_Common(temp_gt))
        img_statics[idx[0],idx[1]] = temp_label
    return img_statics

def ocnn_pro(img_seg, width, height, pro_large_im):
    # Convert the segment indices from 1...n
    img_statics = np.zeros((width, height,pro_large_im.shape[2]),dtype=np.int)
    temp_unique = np.unique(img_seg)
    for i in xrange(pro_large_im.shape[2]):
        for p in xrange(len(temp_unique)):
            temp_value = temp_unique[p]
            idx = np.where(img_seg==temp_value)
            temp_gt = (pro_large_im[idx[0],idx[1],i]).squeeze()
            temp_pro = np.int(Most_Common(temp_gt).flatten())
            img_statics[idx[0],idx[1],i] = temp_pro
    return img_statics

#load training samples from pkl files
training_samples_path = os.path.join(os.getcwd(), 'samples_data/training_patches_ori3.ckpt')
training_labels_path = os.path.join(os.getcwd(), 'samples_data/training_labels_ori3.ckpt')
test_samples_path = os.path.join(os.getcwd(), 'samples_data/test_patches_ori3.ckpt')
test_labels_path = os.path.join(os.getcwd(), 'samples_data/test_labels_ori3.ckpt')

pkl_file_1 = open(training_samples_path, 'rb')
pkl_file_2 = open(training_labels_path, 'rb')
pkl_file_3 = open(test_samples_path, 'rb')
pkl_file_4 = open(test_labels_path, 'rb')

trX = pickle.load(pkl_file_1)
trY_ori = pickle.load(pkl_file_2)
teX = pickle.load(pkl_file_3)
teY_ori = pickle.load(pkl_file_4)

pkl_file_1.close()
pkl_file_2.close()
pkl_file_3.close()
pkl_file_4.close()


#shuffle training and test data with random indices
train_indices = np.arange(len(trX))
np.random.shuffle(train_indices)
trX = trX[train_indices]

test_indices = np.arange(len(teX))
np.random.shuffle(test_indices)
teX = teX[test_indices]
#Convert labels into one-hot style,labels should as 1,2,3,4...
trY = np.zeros((len(trY_ori), len(np.unique(trY_ori)))).astype('int')
ind_tr = np.asarray(trY_ori)-1
trY[np.arange(trY.shape[0]),ind_tr] = 1
#Shuffle training labels
trY = trY[train_indices]

teY = np.zeros((len(teY_ori), len(np.unique(teY_ori)))).astype('int')
ind_te = np.asarray(teY_ori)-1
teY[np.arange(teY.shape[0]),ind_te] = 1
#shuffle test labels
teY = teY[test_indices]


X = tf.placeholder("float", [None, 28, 28, 3])
Y = tf.placeholder("float", [None, 2]) # Number of classes should be defined, here is 2

w = init_weights([5, 5, 3, 100])       # 
w2 = init_weights([3, 3, 100, 300])     # 
w3 = init_weights([5, 5, 300, 500])    # 
w_o = init_weights([500, 2])         # Also number of classes should be defined, here is 2

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = tf.nn.softmax(CNNs.model_28_plain(X, w, w2, w3, w_o, p_keep_conv, p_keep_hidden))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer(0.0001).minimize(cost) # Learning rate 
predict_op = tf.argmax(py_x, 1)
probabilities=py_x

# Set FLAGS
FLAGS.train = 2 # set 1 to train and 2 to predict
FLAGS.checkpoint_dir =os.path.join(os.getcwd())+'/'

# Launch the graph in a session
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # you need to initialize all variables
    
    if FLAGS.train==1:
          # Create a saver.
        init_op = tf.initialize_all_variables().run()
        saver = tf.train.Saver(init_op)
        for step in range(1, max_steps):
            epoch_loss = 0
            training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                _,loss_value = sess.run([train_op, cost], feed_dict={X: trX[start:end], Y: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})
                epoch_loss += loss_value
            print 'loss = %s' % epoch_loss
            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            print(step, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, 
                                feed_dict={X: teX[test_indices], p_keep_conv: 1.0,p_keep_hidden: 1.0})))
                # Save the model checkpoint periodically.
            if step % 9 == 0 or (step + 1) == max_steps:
                saver.save(sess, FLAGS.checkpoint_dir+'saved_model', global_step=step)
                print('Model saved!')
        print('Model saved!')
        
        
            
    elif FLAGS.train==2:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            predictions = []
            print('Begin loading pre-trained model!')
            img = read_tif_as_array(image_path)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restored!')
            [batch_x,rows,cols] = samples.Pre_patches(img,patchshape,resolution)
            test_batch = zip(range(0, len(batch_x), test_batch_size),
                             range(test_batch_size, len(batch_x)+1, test_batch_size))
            pre_count = 1
            for start, end in test_batch:
                prediction = predict_op.eval(feed_dict={X: batch_x[start:end],
                                                        p_keep_conv: 1.0,p_keep_hidden: 1.0}, session=sess)
                
                predictions.append(prediction[0])

            rows = np.array(rows)
            cols = np.array(cols)
            pre_im = np.zeros((max(rows),max(cols)))
            pre_im [np.array(rows)-1,np.array(cols)-1] = predictions
            pre_large_im = cv2.resize(pre_im, (img.shape[1],img.shape[0]),interpolation=0)
            plt.imshow(pre_large_im)
            im_predicted = Image.fromarray(pre_large_im)
            im_predicted.save('PCNN.tif')# This is Pixel-based CNN
            plt.savefig("PCNN.png",dpi=300)
            # Begain object-based CNN classification
            grid = gaussian_grid(sigma)
            if img.shape[2] == 3:
                r = filter_image(img[:,:,0], grid)
                g = filter_image(img[:,:,1], grid)
                b = filter_image(img[:,:,2], grid)
                smooth = (r, g, b)
                diff = diff_rgb
            elif img.shape[2] > 3:
                print("Not supported more than 3 bands.")
            else:
                smooth = filter_image(img, grid)
                diff = diff_grey
            graph = build_graph(smooth, img.shape[0], img.shape[1], diff, neighbor == 8)
            forest = segment_graph(graph, img.shape[0]*img.shape[1], K, min_size, threshold)
            image_seg = generate_image(forest, img.shape[0], img.shape[1])
            ocnn_labels = ocnn(image_seg, img.shape[0], img.shape[1], pre_large_im)
            plt.imshow(ocnn_labels)
            plt.savefig("OCNN.png",dpi=300)
            #sess.close()
        else: print "no pre-trained model found!"
        
tf.reset_default_graph()# clear all stored paramters
