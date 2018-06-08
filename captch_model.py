

import numpy as np
import gzip
import tensorflow as tf
import os


from tensorflow.contrib.data import Dataset, Iterator


class data_set(object):

    def __init__(self):
        self.letters = "abcdefghijklmnopqrstuvwxyz"

        #Training data for individual characters
        self.training_data = []
        self.training_data_labels = []

        #Testing data for individual characters
        self.testing_data = []
        self.testing_data_labels = []

        #Testing data for sliced captchas
        self.captcha_data = []
        self.captcha_data_labels = []

        #load in label values and file paths
        self.load_testing_data()
        self.load_training_data()
        self.load_captcha_data()

    


        self.training_data = tf.constant(self.training_data)
        self.training_data_labels = tf.constant(self.training_data_labels)

        self.testing_data = tf.constant(self.testing_data)
        self.testing_data_labels = tf.constant(self.testing_data_labels)

        self.captcha_data = tf.constant(self.captcha_data)
        self.captcha_data_labels = tf.constant(self.captcha_data_labels)
 
            

        #Creating Tensorflow DataSet objects, set labels to one-hot vectors
        self.training_dataset = Dataset.from_tensor_slices((self.training_data, self.training_data_labels))
        self.testing_dataset = Dataset.from_tensor_slices((self.testing_data, self.testing_data_labels))
        self.captcha_dataset = Dataset.from_tensor_slices((self.captcha_data, self.captcha_data_labels))

        #Shuffle (Don't shuffle captcha data
        self.training_dataset = self.training_dataset.shuffle(10000)
        self.testing_dataset = self.testing_dataset.shuffle(10000)

        #Read in files and decode jpegs/pngs
        self.training_dataset = self.training_dataset.map(self.input_parser)
        self.testing_dataset = self.testing_dataset.map(self.input_parser)
        self.captcha_dataset = self.captcha_dataset.map(self.input_parser2)

        #batch baby batch
        self.training_dataset = self.training_dataset.batch(1000)
        self.testing_dataset = self.testing_dataset.batch(260)
        self.captcha_dataset = self.captcha_dataset.batch(5000) #200 captchas at a time

        #Create iterators for datasets
        self.testing_iterator = self.testing_dataset.make_initializable_iterator()
        self.training_iterator = self.training_dataset.make_initializable_iterator()
        self.captcha_iterator = self.captcha_dataset.make_initializable_iterator()
        

        #Initializers
        self.training_init_op = self.training_iterator.make_initializer(self.training_dataset)
        self.testing_init_op = self.testing_iterator.make_initializer(self.testing_dataset)
        self.captcha_init_op = self.captcha_iterator.make_initializer(self.captcha_dataset)

        #Next elems
        self.next_training_element = self.training_iterator.get_next()
        self.next_testing_element = self.testing_iterator.get_next()
        self.next_captcha_element = self.captcha_iterator.get_next()



    def reshuffle(self):
        #unbatch
        self.training_dataset = self.training_dataset.apply(unbatch())
        #shuffle
        self.training_dataset = self.training_dataset.shuffle(10000)
        #rebatch
        self.training_dataset = self.training_dataset.batch(1000)
        #Remake iterators
        self.training_iterator = self.training_dataset.make_initializable_iterator()
        #Remake iterator_init ops
        self.training_init_op = self.training_iterator.make_initializer(self.training_dataset)
        
    def training_batch(self, batch_size):
        return self.training_dataset.batch(batch_size)
        

    def testing_batch(self, batch_size):
        return self.testing_dataset.batch(batch_size)

    def captcha_batch(self, batch_size):
        return self.captcha_dataset.batch(batch_size)

    def load_training_data(self):
        cwd = os.getcwd()
        for letter in self.letters:
            for num in range(1000):
                self.training_data.append(cwd + "/test3/" + letter +
                                          "/" + "test" + str(num) + ".jpg")
                self.training_data_labels.append(ord(letter) - 97)

    def load_testing_data(self):
        cwd = os.getcwd()
        for letter in self.letters:
            for num in range(10):
                self.testing_data.append(cwd + "/test4/" + letter +
                                          "/" + "test" + str(num) + ".jpg")
                self.testing_data_labels.append(ord(letter) - 97)



    def load_captcha_data(self):
        #Load captcha slice keys into the label list
        cwd = os.getcwd()
        kv = self.read_captcha_kv_store(os.getcwd() + "/keys.txt")
        for num in range(len(kv) - 1):
            k,v = kv[num]
            v = list(v)
            for i in v: # Add on one hot values for labels
                self.captcha_data_labels.append(ord(i) - 97)
            for x in range(1,6): #add on file paths for slices
                self.captcha_data.append(cwd +
                                         "/test/" +
                                         "slice_test" +
                                         str(num) +
                                         "_" +
                                         str(x) +
                                         ".png")
                                    

    
    
    def input_parser(self, img_path, label):
        #convert label
        one_hot = tf.one_hot(label, 26)

        img_file = tf.read_file(img_path)
        img_decode = tf.image.decode_jpeg(img_file, channels = 1)
        img_decode = tf.image.convert_image_dtype(img_decode, tf.float32)
        img_decode = tf.image.resize_images(img_decode, [70, 70])
        img_decode  = tf.reshape(img_decode,[ 70 * 70])

        return img_decode, one_hot


    def input_parser2(self, img_path, label):
        #convert label
        one_hot = tf.one_hot(label, 26)

        img_file = tf.read_file(img_path)
        img_decode = tf.image.decode_png(img_file, channels = 1)
        img_decode = tf.image.convert_image_dtype(img_decode, tf.float32)
        img_decode = tf.image.resize_images(img_decode, [70, 70])
        img_decode  = tf.reshape(img_decode,[ 70 * 70])

        return img_decode, one_hot                                       


    
    #REad k,v filestore
    def read_captcha_kv_store(self, path):
        keystore = open(path, 'r').read()
        records = keystore.split("\n")
        result = []
        for record in records:
            result.append(record.split(','))
        return result




