import tensorflow as tf
import numpy as np
import pickle

def sample_batch(data, all_labels, size_batch, i):
    start = (i * size_batch) % len(data)
    end = (i * size_batch + size_batch) % len(data)

    if not start <= end:
        return data[start:end], all_labels[start:end]
    else:
        data_in_batch = np.vstack((data[start:], data[:end]))
        assert isinstance(all_labels, object)
        labels_in_batch = np.vstack((all_labels[start:], all_labels[:end]))

        return data_in_batch, labels_in_batch

if __name__ == "__main__":

    # adjustable parameters
    learn_rate = 0.001
    max_iterations = 10000
    disp_step = 1
    training_size = 700
    batch_size = 64

    # parameters for cnn
    input_size = 599 * 13 * 5
    dropout_rate = 0.72
    num_classes = 10

    sound_data = []
    all_labels = []

    # reads from files that were created using the preprocessing scripts (mfcc saver)
    with open('data', 'r') as f:
        info = f.read()
        sound_data = pickle.loads(info)

    assert isinstance(sound_data, object)
    sound_data = np.asarray(sound_data)
    sound_data = sound_data.reshape((sound_data.shape[0], input_size))

    with open('labels', 'r') as f:
        info = f.read()
        all_labels = pickle.loads(info)

    # shuffle data
    shuffled_data = np.random.permutation(len(sound_data))
    sound_data = sound_data[shuffled_data]
    all_labels = all_labels[shuffled_data]

    # train/test split
    training_X = sound_data[:training_size]
    training_y = all_labels[:training_size]
    testing_X = sound_data[training_size:]
    testing_y = all_labels[training_size:]

    # initialize tensorflow graph
    X = tf.placeholder(tf.float32, [None, input_size])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    prob_dropout = tf.placeholder(tf.float32)

    def max_pooling(sound, k):
        return tf.nn.max_pool(sound, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_layer(song_sample, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(song_sample, w, strides=[1, 1, 1, 1],
                                                      padding='SAME'), b))

    # creates and trains convolutional neural network for music sample classification
    def create_CNN(input_layer, weights_, biases_, dropout_rate):
        # reshape input
        input_layer = tf.reshape(input_layer, shape=[-1, 599, 13, 5])

        # convolution layer w/ max pooling and dropout applied
        conv1 = conv_layer(input_layer, weights_['wc1'], biases_['bc1'])
        conv1 = max_pooling(conv1, k=4)
        conv1 = tf.nn.dropout(conv1, dropout_rate)

        # 2nd convolution layer w/ max pooling and dropout applied
        conv2 = conv_layer(conv1, weights_['wc2'], biases_['bc2'])
        conv2 = max_pooling(conv2, k=2)
        conv2 = tf.nn.dropout(conv2, dropout_rate)

        # dense layer w/ relu activation and dropout applied
        dense1 = tf.reshape(conv2, [-1, weights_['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, weights_['wd1']), biases_['bd1']))
        dense1 = tf.nn.dropout(dense1, dropout_rate)

        output = tf.add(tf.matmul(dense1, weights_['out']), biases_['out'])
        return output


    # store biases and weights for CNN
    biases = {
        'bc1': tf.Variable(tf.random_normal([149])),
        'bc2': tf.Variable(tf.random_normal([73])),
        'bc3': tf.Variable(tf.random_normal([35])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    weights = {
        'wc1': tf.Variable(tf.random_normal([4, 4, 5, 149])),
        'wc2': tf.Variable(tf.random_normal([4, 4, 149, 73])),
        'wc3': tf.Variable(tf.random_normal([2, 2, 73, 35])),
        'wd1': tf.Variable(tf.random_normal([75 * 2 * 73, 1024])),
        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }

    # create model
    model = create_CNN(X, weights, biases, prob_dropout)

    # loss and optimizer (softmax w/ cross entropy and adam, as usual haha)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

    # evaluate model
    predicted_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    _accuracy = tf.reduce_mean(tf.cast(predicted_correct, tf.float32))

    restart = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # launch graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(restart)
        step = 1
        # train until max iterations is reached
        while step * batch_size < max_iterations:
            batch_xs, batch_ys = sample_batch(training_X, training_y, batch_size, step)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, prob_dropout: dropout_rate})

            if step % disp_step == 0:
                accuracy = sess.run(_accuracy, feed_dict={X: batch_xs, Y: batch_ys, prob_dropout: 1.})
                loss = sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, prob_dropout: 1.})
                print("iteration " + str(step * batch_size) + ", loss for batch = " + \
                      "{:.6f}".format(loss) + ", accuracy= " + "{:.5f}".format(accuracy))

                saved = saver.save(sess, "model.ckpt")
                print("model saved in file: %s" % saved)
            step += 1
        print("model trained!")

        saved = saver.save(sess, "model.pkt")
        print("model saved as: %s" % saved)
        print("accuracy:", sess.run(_accuracy, feed_dict={X: testing_X,
                                                          Y: testing_y,
                                                          prob_dropout: 1.}))
