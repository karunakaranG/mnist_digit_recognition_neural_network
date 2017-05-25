import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from preprocess_model import preprocessed_image
from slidewindow import main_slide
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

print(len(mnist.test.images))
print(len(mnist.train.images))

"""
inputdata > weight and bias > hidden layer1(activation function) > weights > hidden layer2(activation function) > weights > output layer

optimization function---->minimize cost function ->>>SGD,ADAM, ADA grad

Backpropagation
modify weights according to cost function

feedforward+backward propagation=epoach(10,20,30)

"""

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500


n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 625])
y = tf.placeholder('float')


def neural_network_model(data):
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([625, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    #hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
    #                  'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    #l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    #l4 = tf.nn.relu(l4)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output
def preprocess_train(xdataset):
    x_data, y_data=xdataset

    final_x=[]
    
    for some_x in x_data:
        first= np.array((some_x*255),dtype=np.uint8)
        data1 = first.reshape(28,28)
         
        final=preprocessed_image(data1)
        final_x.append(final)
        
        
    
    return final_x, y_data

def preprocess_test(xdataset):
    x_data=xdataset

    final_x=[]
    
    for some_x in x_data:
        first= np.array((some_x*255),dtype=np.uint8)
        data1 = first.reshape(28,28)
         
        final=preprocessed_image(data1)
        final_x.append(final)
        
        
    
    return final_x

def train_neural_network(x):
    prediction = neural_network_model(x)
   
    #cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y) )
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(cost)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = preprocess_train(mnist.train.next_batch(batch_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        some_val=preprocess_test(mnist.test.images)
        
        print('Accuracy:',accuracy.eval({x:some_val, y:mnist.test.labels}))
        # i=10
        #while i<20:
        #    s1=mnist.test.images[i]
        #    data1=s1.reshape(28,28)
        #    plt.imshow(data1,cmap='gray')
        #    plt.show()
        #    i=i+1
            
        #x1 = mnist.test.images[23]
        #first_image = np.array(x1)
        #data = first_image.reshape(28,28)
        #cv2.imwrite('imageq.jpg',(data*255))
        #plt.imshow(data, cmap='gray')
        #plt.show()
        
#--------------------------------------------------
        img=cv2.imread('manual_test_image.jpg',0)
        img_array=np.array(img,dtype=np.uint8)
        final_c=main_slide(img_array)
        for val in final_c:
            result = sess.run(tf.argmax(prediction,1), feed_dict={x: [val]})
            print(result)

       
       
        
train_neural_network(x)



