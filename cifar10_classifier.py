from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from datetime import datetime
import time, sys, select, math, cPickle, pickle, tarfile
import tensorflow as tf, numpy as np
from ConvNet import ConvNet
from cifar10_input_array import load

MOVING_AVG_DECAY = 0.99
NUM_STEPS_PER_DECAY = 100000
LEARNING_RATE_DECAY_FACTOR = 0.5
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.9
GRADIENT_CLIP_AVG_NORM = 0.2

DROPOUT = 0.75

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('training_num', 50000, """training number""")
flags.DEFINE_integer('valid_num', 10000, """validation number""")
flags.DEFINE_integer('batch_size', 128, """Batch size""")
flags.DEFINE_integer('val_batch_size', 64, """Validation batch size""")
flags.DEFINE_integer('max_epoch', 100, """Max epochs to run trainer""")
flags.DEFINE_integer('display_interval', 1, """interval for display""")
flags.DEFINE_integer('val_interval', 1, """interval for validation""")
flags.DEFINE_integer('summary_interval', 1, """interval for summary""")
flags.DEFINE_string('save_directory', '../results/0.75_en',
    """directory to save""")
flags.DEFINE_string('checkpoint', None,
    """if sets, resume training on the checkpoint""")

def loss_and_accuracy(logits, labels, phase):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name = 'cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection(phase+'_losses', cross_entropy_mean)

    op = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.to_float(op), name = 'accuracy')
    weight_losses = tf.add_n(tf.get_collection('weight_losses'),
            name = 'weight_losses')
    return tf.add(cross_entropy_mean, weight_losses, name = 'total_losses'), cross_entropy_mean, accuracy

def _add_loss_accuracy_summary(total_loss, cross_entropy_loss, accuracy,
        phase):

    if phase == 'train': moving_avg_decay = MOVING_AVG_DECAY
    else: moving_avg_decay = 0.9

    averages = tf.train.ExponentialMovingAverage(moving_avg_decay,
                        name = 'averages')
    losses = tf.get_collection(phase+'_losses')
    loss_and_accs= losses + [total_loss, accuracy]
    
    if phase == 'train':
        weight_losses = tf.get_collection('weight_losses')
        loss_and_accs += weight_losses

    averages_op = averages.apply(loss_and_accs)
    for l in loss_and_accs:
        tf.scalar_summary(phase+'_'+l.op.name, l)
        tf.scalar_summary(phase+'_'+l.op.name + ' (avg)', averages.average(l))
    
    return averages_op


def train(total_loss, lr, lr_decay, global_epoch):
    tf.scalar_summary('learning_rate', lr)
    opt = tf.train.RMSPropOptimizer(lr, lr_decay)
    # TODO

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    train_op = opt.minimize(total_loss)
    
    return train_op

def run_training():

    trainX, trainY, validX, validY = load(one_hot = False, cv = -1)

    mean = []
    for shape in [trainX.shape, validX.shape]:
        mean.append(np.repeat(np.mean(trainX.transpose([3,0,1,2]).reshape(3,-1),axis=1),shape[0]*shape[1]*shape[2]).reshape(3,-1).transpose().reshape(shape[0],shape[1],shape[2],shape[3]))
        
    trainX = trainX - mean[0]
    validX = validX - mean[1]
    #testX = testX - mean[2]
    

    with tf.Graph().as_default():
        
        convNet = ConvNet(dropout = DROPOUT)

        global_epoch = tf.Variable(0, trainable=False)
        learning_rate = tf.Variable(INITIAL_LEARNING_RATE, trainable=False)
        learning_rate_decay = tf.Variable(LEARNING_RATE_DECAY, trainable=False)
        train_mode = tf.Variable(True, trainable=False)

        X = tf.placeholder("float", [FLAGS.batch_size, 28, 28, 3])
        Y = tf.placeholder("float", [FLAGS.batch_size])
        phase = tf.placeholder("string")
                 
        logits = convNet.inference(X, FLAGS.batch_size, train_mode)
        
        loss, cross_entropy_loss, accuracy = loss_and_accuracy(
                logits, Y, phase=phase)
        avg_op = _add_loss_accuracy_summary(loss, cross_entropy_loss,
                accuracy, phase=phase)
        
        with tf.control_dependencies([avg_op]):
            valid_op = tf.no_op(name='validation')
            valid_loss_op = cross_entropy_loss
            valid_acc_op = accuracy
            valid_error_op = (1.-accuracy)*100.
            
        with tf.control_dependencies([avg_op]):
            train_op = train(cross_entropy_loss, learning_rate,
                    learning_rate_decay, global_epoch)
            loss_op = cross_entropy_loss
            accuracy_op = accuracy
            error_op = (1.-accuracy)*100.
            learning_rate_op = learning_rate
        
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(tf.all_variables(),
                keep_checkpoint_every_n_hours=12.0)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            if FLAGS.checkpoint is not None:
                saver.restore(sess=sess, save_path = FLAGS.checkpoint)
            else:
                sess.run(init)

            tf.train.start_queue_runners(sess=sess)
            current_epoch = global_epoch.eval(sess)

            summary_writer = tf.train.SummaryWriter(
                    FLAGS.save_directory,
                    graph_def = sess.graph_def)

            best_error = 100.0
            bad_cnt = 0

            print ("*** start training ***")
            
            for epoch in xrange(current_epoch, FLAGS.max_epoch):
                start_time = time.time()
                
                training_steps = int(FLAGS.training_num / FLAGS.batch_size)
                shuffler = np.random.permutation(FLAGS.training_num)
                trainX = trainX[shuffler]
                trainY = trainY[shuffler] 
                
                for step in xrange(training_steps):
                    start = step * FLAGS.batch_size
                    end = start + FLAGS.batch_size
                    thisX = trainX[start:end]
                    # random flip left right
                    if(np.random.randint(2)==0):
                        thisX = thisX[:,:,::-1,:]
                    # random crop to (28, 28)
                    ind = np.random.randint(4)
                    thisX = thisX[:,ind:ind+28,ind:ind+28,:]
                    # random gray with 20% prob
                    if(np.random.randint(5)==0):
                        thisX = np.repeat(
                            np.average(thisX,axis=3),3).reshape(-1,28,28,3)
                    thisY = trainY[start:end]
                    if step == training_steps - 1:
                        _, lr, loss, error, summary_string = sess.run(
                            [train_op, learning_rate_op, loss_op,
                                error_op, summary_op],
                            feed_dict={X:thisX, Y:thisY, phase:'train'})
                    else:
                        _, lr, loss, error = sess.run(
                            [train_op, learning_rate_op, loss_op, error_op],
                            feed_dict={X:thisX, Y:thisY, phase:'train'})
                
                t = time.time()-start_time
                examples_per_sec = FLAGS.batch_size * training_steps / t
                print ('epoch %d : lr = %.1e, loss = %.4f, error = %.2f'
                        '(%.1f examples/sec)'
                        % (epoch, lr, loss, error, examples_per_sec))
                summary_writer.add_summary(summary_string, epoch)
                
                # eval validation accuracy
                if epoch % FLAGS.val_interval == 0:
                    val_loss_sum = 0.0
                    val_error_sum = 0.0
                    valid_steps = int(FLAGS.valid_num / FLAGS.batch_size)
                    sess.run(train_mode.assign(False))
                    
                    start_time = time.time()
                    for v_step in xrange(valid_steps):
                        start = v_step * FLAGS.batch_size
                        end = start + FLAGS.batch_size
                        thisX = validX[start:end,2:30,2:30,:]
                        thisY = validY[start:end]
                        
                        if v_step == valid_steps-1:
                            _,val_loss,val_error,summary_string = sess.run(
                                    [valid_op, valid_loss_op, valid_error_op,
                                        summary_op],
                                    feed_dict = {X:thisX, Y:thisY, phase:'valid'})
                        else:
                            _, val_loss, val_error = sess.run(
                                [valid_op, valid_loss_op, valid_error_op],
                                feed_dict = {X:thisX, Y:thisY, phase:'valid'})
                        val_loss_sum += val_loss
                        val_error_sum += val_error

                    val_loss = val_loss_sum / valid_steps
                    val_error = val_error_sum / valid_steps
                    t = time.time() - start_time
                    examples_per_sec = FLAGS.batch_size * valid_steps / t

                    print ('[Valid] epoch %d : loss = %.4f, error = %.2f'
                            '(%.1f examples/sec)'
                            %(epoch, val_loss, val_error, examples_per_sec))
                    
                    summary_writer.add_summary(summary_string, epoch)
                    sess.run(train_mode.assign(True))
                                
                if best_error > val_error:
                    best_error = val_error
                    bad_cnt = 0
                else:
                    bad_cnt += 1
                    if bad_cnt > 5:
                        sess.run(learning_rate.assign(learning_rate*0.1))

                if epoch % 10 == 0:
                    saver.save(sess, FLAGS.save_directory + '/model.ckpt',
                            global_step = step)


def main(argv=None):
    run_training()

if __name__ == '__main__':
    tf.app.run()


