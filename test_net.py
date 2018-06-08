import tensorflow as tf
from captch_model import data_set
import os
import time
from PIL import Image, ImageChops
import math
import PIL


#Timing Metric
start_time = time.time()

#Consider larger datasets for a new training batch

data = data_set()

n_nodes_hl1 = 1300
n_nodes_hl2 = 1300
n_nodes_hl3 = 1300
n_nodes_hl4 = 1300
n_nodes_hl5 = 1300
n_nodes_hl6 = 1300

#3 hl at 2450 each = 72% accurracy
#6 hl at 1000 each = 73%
#6 hl at 1300 each = 88.8%
#6 hl at 1300 each = 93.8% more epochs 25
#6 hl at 1300 each = ~96 % accuracy  40 epochs



n_classes = 26
training_batch_size = 100
testing_batch_size = 50
cwd = os.getcwd()




#heihgt and width

x = tf.placeholder('float', [ None, 70 * 70 * 1]) # 35 * 35 each images might need to mult by 3
y = tf.placeholder('float') #labels are numbers, equal to a value in range[0,25]


#Define the neural network model, and set up graph
def neural_network_model(data):

    
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([70 * 70 * 1, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_layer_4 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    
    hidden_layer_5 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}

    
    hidden_layer_6 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl6]))}

    output_layer   = {'weights': tf.Variable(tf.random_normal([n_nodes_hl6, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    #(input data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']),  hidden_layer_1['biases'])

    #activation function
    l1 = tf.nn.relu(l1) 

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
#
    l4 = tf.add(tf.matmul(l3, hidden_layer_4['weights']), hidden_layer_4['biases'])
    l4 = tf.nn.relu(l4)
    
    l5 = tf.add(tf.matmul(l4, hidden_layer_5['weights']), hidden_layer_5['biases'])
    l5 = tf.nn.relu(l5)
    
    l6 = tf.add(tf.matmul(l5, hidden_layer_6['weights']), hidden_layer_6['biases'])
    l6 = tf.nn.relu(l6)
#
    output = tf.matmul(l6, output_layer['weights']) + output_layer['biases']
    return output




#Tell TF in session how to train model
def train_neural_network(x):
    #Takes the input data, runs through computation graph and determines a prediciton
    prediction = neural_network_model(x)
    
    #Difference between model prediction and actual value
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))

    #Minimize the cost, def learning_rate = .001
    #Try a different rate
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #Feed Forward + back-prop  = 1 epoch
    num_epochs = 40
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
                 
        sess.run(data.testing_init_op)
        sess.run(data.training_init_op)
        sess.run(data.captcha_init_op)
        

        for epoch in range(num_epochs):
            epoch_cost = 0
            sess.run(data.training_init_op)
            sess.run(data.testing_init_op)
            sess.run(data.captcha_init_op)
        
            for _ in range(26):
                x_, y_ = sess.run(data.next_training_element)
                _, c = sess.run([optimizer, cost], feed_dict ={x: x_, y: y_})
                epoch_cost +=c
            print('Epoch', epoch, 'completed out of', num_epochs, 'loss', epoch_cost)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print("(Correct, Accuracy): ", correct, accuracy)
            x__, y__ = sess.run(data.next_captcha_element)
            print("Accuracy:", accuracy.eval({x: x__ , y: y__}), "\n\n")
        

            
           
train_neural_network(x)
finish_time = time.time()
print((finish_time - start_time)/60, "minutes training time")


    

    
