import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import sys
sys.path.insert(0, '/home/long/Desktop/Final_Project/feats/Functions')
from cnn_util import * 

class Food_classification():

    def __init__(self, n_words, batch_size):
 	self.n_words = n_words
	self.batch_size = batch_size
	#self.Wa = tf.Variable(tf.random_uniform([512, 256], -1.0, 1.0), name='Wa')
	#self.We = tf.Variable(tf.random_uniform([256, 256], -1.0, 1.0), name='We')
        self.Wemb = tf.Variable(tf.random_uniform([n_words, 512], -1.0, 1.0), name='Wemb')
	#self.Va = tf.Variable(tf.random_uniform([256, 1], -1.0, 1.0), name='Va')
	self.decode_w = tf.Variable(tf.random_uniform([n_words, 512], -1.0, 1.0), name = 'decode_w')

    def build_model(self):
        context = tf.placeholder("float32", [self.batch_size, 196, 512])
        labels = tf.placeholder("float32", [self.batch_size, self.n_words])
	
	context_flat = tf.reshape(context, [-1, 512])

	loss = 0.0
	logits = []

        for ind in range(self.n_words):
	    word_emb = tf.reshape(self.Wemb[ind,:], [1,512])  
	    e_t = tf.matmul(context_flat, word_emb)
	    #e_t = tf.matmul(context_flat, self.Wa) + tf.matmul(word_emb, self.We)
	    e_t = tf.nn.tanh(e_t)
	    #e_t = tf.matmul(e_t, self.Va)
	    e_t = tf.reshape(e_t, [self.batch_size, 196])

	    alpha = tf.nn.softmax(e_t)

	    weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)

	    logit_value = tf.matmul(weighted_context, tf.reshape(self.decode_w[ind,:], [512, 1])) 
	    logits.append(logit_value)

        logits = tf.transpose(tf.reshape(logits, [self.n_words, self.batch_size]))

	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
        loss = tf.reduce_sum(loss)/self.batch_size

	return context, labels, loss, self.Wemb

    def build_generator(self):
	context = tf.placeholder("float32", [1, 196, 512])
	context_flat = tf.squeeze(context)

	alpha_list=[]
	logits = []

        for ind in range(self.n_words):
	    word_emb = tf.reshape(self.Wemb[ind,:], [512, 1])  

	    #e_t = tf.matmul(context_flat, self.Wa) + tf.matmul(word_emb, self.We)
	    e_t = tf.matmul(context_flat, word_emb)
	    e_t = tf.nn.tanh(e_t)
	    #e_t = tf.matmul(e_t, self.Va)
	    e_t = tf.reshape(e_t, [1, 196])

	    alpha = tf.nn.softmax(e_t)
	    alpha_list.append(alpha)

            weighted_context = tf.reduce_sum(context_flat * tf.reshape(alpha, [196, 1]), 0)
	    weighted_context = tf.reshape(weighted_context,[1,512])

	    logit_value = tf.matmul(weighted_context, tf.reshape(self.decode_w[ind,:], [512, 1])) 
	    logits.append(logit_value)

	return context, alpha_list, tf.nn.sigmoid(logits)


######  Parameters ##########
n_epochs= 2000
batch_size = 64
n_words = 7
learning_rate = 0.001
#############################
###### Parameters ###########
annotation_path = '/home/long/Desktop/Final_Project/feats/labels_y.npy'
feat_path = '/home/long/Desktop/Final_Project/feats/googlefood.npy'
model_path = '/home/long/Desktop/Final_Project/model/'
#############################


def train(): 
    caption = np.load(annotation_path)
    feats = np.load(feat_path)

    sess = tf.InteractiveSession()

    classification = Food_classification(n_words, batch_size)


    context, labels, loss, logits = classification.build_model()
    saver = tf.train.Saver(max_to_keep=1500)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.initialize_all_variables().run()

    epochs = []
    losss = []

    for epoch in range(n_epochs):
	index = np.arange(len(caption))
	np.random.shuffle(index)

        for start, end in zip( \
                range(0, len(caption), batch_size),
                range(batch_size,len(caption), batch_size)):

	    current_feats = feats[index[start:end]].reshape(-1, 512, 196).swapaxes(1,2)
            current_caption = caption[index[start:end]]

            _, loss_value, logit = sess.run([train_op, loss, logits], feed_dict={
                context:current_feats,
                labels:current_caption
                })
	saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
	
	if epoch %20 == 0:
            print logit[0]
	###### Plot ########
	epochs.append(epoch)
	losss.append(loss_value)
	print epoch, loss_value
    plt.plot(epochs, losss)
    plt.ylim(0,10)
    plt.title('e_t = ai*Wemb')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
	#####################



def test(test_feat=None, model_used=None):
    feat = np.load(test_feat).reshape(-1, 512, 196).swapaxes(1,2)

    sess = tf.InteractiveSession()

    classification = Food_classification(n_words, batch_size)

    context, alpha_list, logit_list = classification.build_generator()
    saver = tf.train.Saver()
    saver.restore(sess, model_used)

    alpha, logit = sess.run([alpha_list, logit_list], feed_dict={context:feat})

    return alpha, logit

############### test parameter #############
test_feat_path = "/home/long/Desktop/Food_caption/Less_data/testing/feat/4.npy"
test_image = "/home/long/Desktop/Food_caption/Less_data/testing/image/4.jpeg"
model_used = "/home/long/Desktop/Final_Project/model/model-4598"

def show_img():

    alpha, logit = test(test_feat_path, model_used)

    img = crop_image(test_image)

    alphas = np.array(alpha)
    print alphas[0,0,:] 
    n_words = alphas.shape[0] 
    w = np.round(np.sqrt(n_words))
    h = np.ceil(np.float32(n_words) / w)

    plt.subplot(w, h, 1)
    plt.imshow(img)
    plt.axis('off')

    smooth = False
    label = ['salmon','asparagus','eggs',1,1,1,1]
    for ii in xrange(n_words):
        plt.subplot(w, h, ii+2)
        lab = [ii]
    
        plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, lab, color='black', fontsize=13)
        plt.imshow(img)
    
        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alphas[ii].reshape(14,14), upscale=16, sigma=20)
        else:
            alpha_img = skimage.transform.resize(alphas[ii].reshape(14,14), [img.shape[0], img.shape[1]])
        
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

#train()
show_img()
