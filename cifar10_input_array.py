import cPickle, pickle, sys, tarfile
import numpy as np

def load(cv = 0, one_hot = True):
    """
    args:
        cv : cross-validation number ( 0 ~ 4 )
        one_hot : boolean
    returns:
        trainX, trainY, validX, validY, testX, testY : numpy array and label
    """

    f1 = tarfile.open("../data/cifar-10-python.tar.gz",'r:gz')
    training_set=[]
    for i in xrange(1,6):
        training_set.append(cPickle.load(f1.extractfile('cifar-10-batches-py/data_batch_%d' % (i))))
    test_set = cPickle.load(f1.extractfile('cifar-10-batches-py/test_batch'))
    
    def to_one_hot(y):
        Y = np.zeros((len(y), 10))
        for i in range(len(y)):
            k = y[i]
            Y[i][k]=1
        return Y
    
    tot_trainX = [np.transpose(data['data'].reshape(-1,3,32,32),[0,2,3,1])
                        for data in training_set]
    if one_hot:
        tot_trainY = [to_one_hot(data['labels']) for data in training_set]
    else:
        tot_trainY = [data['labels'] for data in training_set]
    testX = np.transpose(
            test_set['data'].reshape(-1,3,32,32),[0,2,3,1])
    if one_hot:
        testY = to_one_hot(test_set['labels'])
    else:
        testY = test_set['labels']
    
    
    concate = False
    for i in range(5):
        if i==cv:
            validX = tot_trainX[i]
            validY = tot_trainY[i]
        elif concate:
            trainX = np.concatenate((trainX,
                tot_trainX[i]))
            trainY = np.concatenate((trainY,
                tot_trainY[i]))
        else:
            trainX = tot_trainX[i]
            trainY = tot_trainY[i]
            concate = True
    if cv==-1: return trainX, trainY, testX, testY
    return trainX, trainY, validX, validY, testX, testY



