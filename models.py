import torch
import torch.nn.functional as F
import torch.utils.data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Conv2D, \
                                    MaxPooling1D, GRU, Bidirectional, Flatten, ReLU, \
                                    Reshape, Multiply
import h5py
import numpy as np
import matplotlib
matplotlib.use('agg') # different backend so we don't need tkinter
import matplotlib.pyplot as plt


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def set_params(self, p, lr, stdev, batchnorm_momentum):
        pass
    
    def get_params(self):
        return None
    
    def printParams(self):
        print("input size", self.input_dim)
        print("output size", self.output_dim)
    
    def reinit_weights(self):
        self.__init__(self.input_dim, self.output_dim)
        for child in self.modules():
            try:
                torch.nn.init.xavier_uniform_(child.weight)
            except Exception as e:
                pass

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred.reshape(1,list(y_pred.size())[0]) # flatten

class TwoLayerNet(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,p=0.0,regression=False):
        super(TwoLayerNet,self).__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.regression = regression
        self.p = p
        
        self.drop_layer = torch.nn.Dropout(p=p)
        self.layer1 = torch.nn.Linear(input_size,hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size,output_size)
        
    def set_params(self, p, lr, stdev, batchnorm_momentum):
        self.drop_layer = torch.nn.Dropout(p=p)
        
    def get_params(self):
        return self.p, 0, 0

    def printParams(self):
        print("input size", self.input_size)
        print("hidden size", self.hidden_size)
        print("output size", self.output_size)
        
    def reinit_weights(self):
        self.__init__(self.input_size, self.hidden_size, self.output_size, self.p, self.regression)
        for child in self.modules():
            try:
                torch.nn.init.xavier_uniform_(child.weight)
            except Exception as e:
                pass

    def forward(self,input):
        out = self.drop_layer(input)
        out = self.layer1(out)
        out = F.relu(out)
        out = self.layer2(out)
        if self.regression:
            out = 8*torch.sigmoid(out)-4
            return out.reshape(1,list(out.size())[0]) # flatten
        else: # classification
            out = torch.sigmoid(out)
            return out.view(-1) # flatten

class ThreeLayerNet(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,p=0.0,regression=False):
        super(ThreeLayerNet,self).__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.regression = regression
        self.p = p
        
        self.drop_layer = torch.nn.Dropout(p=p)
        self.layer1 = torch.nn.Linear(input_size,hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size,hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size,output_size)
        
    def set_params(self, p, lr, stdev, batchnorm_momentum):
        self.p = p
        self.drop_layer = torch.nn.Dropout(p=p)
        
    def get_params(self):
        return self.p, 0, 0
    
    def printParams(self):
        print("input size", self.input_size)
        print("hidden size", self.hidden_size)
        print("output size", self.output_size)
    
    def reinit_weights(self):
        self.__init__(self.input_size, self.hidden_size, self.output_size, self.p, self.regression)
        for child in self.modules():
            try:
                torch.nn.init.xavier_uniform_(child.weight)
            except Exception as e:
                pass

    def forward(self,input):
        out = self.drop_layer(input)
        out = self.layer1(out)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.layer3(out)
        if self.regression:
            out = 8*torch.sigmoid(out)-4
            return out.reshape(1,list(out.size())[0]) # flatten
        else:
            out = torch.sigmoid(out)
            return out.view(-1) # flatten


class GaussianNoise(torch.nn.Module): # custom module to implement Gaussian random noise in nn.Sequential containers
    def __init__(self, mean, stdev):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.stdev = stdev
        
    def forward(self, ins):
        if self.training:
            noise = ins.data.new(ins.size()).normal_(self.mean, self.stdev)
            return ins + noise
        return ins

class NormaliseCustom (torch.nn.Module):
    def __init__(self):
        super(NormaliseCustom, self).__init__()

    def forward(self, ins):
        means  = torch.mean(ins.data, dim=-1)
        stdevs = torch.std(ins.data,  dim=-1)
        ins = ins- means[:,:,None].repeat_interleave(23, dim=-1)
        ins = ins/stdevs[:,:,None].repeat_interleave(23, dim=-1)
        return ins


class XGBoostTree():
    def __init__(self, colsample=0.7, lr=0.5, max_depth=7, min_child_weight=1):
        self.colsample, self.lr, self.max_depth, self.min_child_weight = colsample, lr, max_depth, min_child_weight
        
    def reinit_weights(self):
        pass # no need to implement this - model is trained from scratch each time anyway, except when xgb_model is set

    def set_params(self, colsample=None, lr=None, max_depth=None, min_child_weight=None):
        if colsample is not None: self.colsample = colsample
        if lr is not None: self.lr = lr
        if max_depth is not None: self.max_depth = max_depth
        if min_child_weight is not None: self.min_child_weight = min_child_weight

    def get_params(self):
        return self.colsample, self.lr, self.max_depth, self.min_child_weight

    def printParams(self):
        print("colsample_bytree", self.colsample)
        print("lr", self.lr)
        print("max_depth", self.max_depth)
        print("min_child_weight", self.min_child_weight)


class ConvolutionalNet(torch.nn.Module):
    def __init__(self, regression, p=0.0, device=None, seqDim=4, epiStart=4, epiDim=10, mean = 0, stdev = 0.254, batchnorm_momentum = 0.1):
        super(ConvolutionalNet, self).__init__()
        self.device = device
        self.seqDim, self.epiStart, self.epiDim = seqDim, epiStart, epiDim
        self.p, self.mean, self.stdev, self.batchnorm_momentum = p, mean, stdev, batchnorm_momentum
        self.regression = regression
        self.BatchNorm = torch.nn.BatchNorm1d(self.seqDim+self.epiDim, momentum=self.batchnorm_momentum)
        self.Normalise = NormaliseCustom()
        self.conjoinedLayer1 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1), # preserve dimensions
            torch.nn.ReLU()
            )
        self.conjoinedLayer2 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU()
            )
        self.conjoinedLinear = torch.nn.Linear(512, 1)
        self.EncodeLinear = torch.nn.Linear(128*2, 128*2)
        
        self.EncodeLayer2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.2)
            )
        self.EncodeLayer3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=0),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.2)
            )

        self.set_params(p, None, stdev, batchnorm_momentum)

    def reinit_weights(self):
        self.__init__(regression=self.regression, p=self.p, device=self.device, seqDim=self.seqDim, epiStart=self.epiStart, epiDim=self.epiDim, mean=self.mean, stdev=self.stdev, batchnorm_momentum=self.batchnorm_momentum)
        for child in self.modules():
            try:
                torch.nn.init.xavier_uniform_(child.weight)
            except Exception as e:
                pass

    def set_params(self, p=None, lr=None, stdev=None, batchnorm_momentum=None): # lr is not needed here but included so we can unpack the full hyperparameter array as arguments
        if p is not None: self.p = p
        if stdev is not None: self.stdev = stdev
        if batchnorm_momentum is not None: self.batchnorm_momentum = batchnorm_momentum

        # set dropout p, Gaussian stdev
        # don't set batchnorm_momentum as this will probably mess up the running stats
        # could inherit these member functions from EncodeConvolutionalNet
        self.EncodeLayer1 = torch.nn.Sequential(
            torch.nn.Conv1d(int(self.seqDim+self.epiDim), 32, kernel_size=3, stride=2, padding=0),
            #nn.BatchNorm1d(32),
            GaussianNoise(self.mean, self.stdev),
            torch.nn.Dropout(p=self.p),
            torch.nn.LeakyReLU(0.2)
            )
        self.EncodeLayer1.to(self.device)
        
        if (self.device != "cpu"):
            for children in self.modules():
                if type(children)==torch.nn.BatchNorm1d:
                    pass#children.track_running_stats = False

    def get_params(self):
        return self.p, self.stdev, self.batchnorm_momentum

    def printParams(self):
        print("batchnorm_momentum", self.batchnorm_momentum)
        print("dropout probability", self.p)
        print("Gaussian noise stdev", self.stdev)

    def forward_encode(self, x):
        out = self.EncodeLayer1(x)
        out = self.EncodeLayer2(out)
        out = self.EncodeLayer3(out)
        latentRepres = out
        return latentRepres

    def forward(self, x):
        # TODO: import pretrained weights from autoencoder for the two separate networks below
        if (type(x) is not torch.Tensor): # arguments can be supplied as lists too
            x = torch.FloatTensor(x).to(self.device)
        x = x.view(x.size(0), self.seqDim+self.epiDim, -1).to(self.device)

        x = self.BatchNorm(x)
        seq_encoding = self.forward_encode(x)

        out = self.conjoinedLayer1(seq_encoding)
        out = self.conjoinedLayer2(out)
        out = out.reshape(out.size(0), -1) # flatten last axis to 512x1
        out = self.conjoinedLinear(out)
        if (not self.regression): out = torch.sigmoid(out) # for classification
        return out

def createDeepCRISPRmodel(regression, device, seqDim, epiStart, epiDim, lossfct='mse', optimiser='adam'):
    model = tf.keras.models.Sequential([
        Conv2D(32, (1, 3), padding='same', name='e_1', input_shape=(1, 23, num_feats), kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_1u'),  # momentum=0.0, scale=False,
        ReLU(),
        Conv2D(64, (1, 3), padding='same', strides=(1, 2), name='e_2', kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_2u'),  # momentum=0.0, scale=False,
        ReLU(),
        Conv2D(64, (1, 3), padding='same', name='e_3', kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_3u'),  # momentum=0.0, scale=False,
        ReLU(),
        Conv2D(256, (1, 3), padding='same', strides=(1, 2), name='e_4', kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_4u'),  # momentum=0.0, scale=False,
        ReLU(),
        Conv2D(256, (1, 3), padding='same', name='e_5', kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_5u'),  # momentum=0.0, scale=False,
        ReLU(),

        Conv2D(512, (1, 3), padding='same', strides=(1, 2), name='e_6', kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_6l'),  # momentum=0.99, center=False, scale=False,
        ReLU(),
        Conv2D(512, (1, 3), padding='same', name='e_7', kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_7l'),  # momentum=0.99, center=False, scale=False,
        ReLU(),
        Conv2D(1024, (1, 3), padding='valid', name='e_8', kernel_initializer='he_uniform'),
        BatchNormalization(name='ebn_8l'),  # momentum=0.99, center=False, scale=False,
        ReLU(),
        Conv2D(1, (1, 1), padding='valid', name='e_9', activation=tf.nn.sigmoid, kernel_initializer='he_uniform')
    ])

    model.compile(loss=lossfct,
                  optimizer=optimiser,
                  metrics=['mae', 'mse'])
    return model

class CRNNCrisprModel(keras.Model):
    def __init__(self, regression, device, seqDim, epiStart, epiDim, numBpWise=0, verbose=False, isHPC=False, sequencesEncodedAsOne=False, **kwargs):
        super(CRNNCrisprModel, self).__init__()
        self.regression, self.device, self.seqDim, self.epiStart, self.epiDim, self.numBpWise = regression, device, seqDim, epiStart, epiDim, numBpWise
        self.sequencesEncodedAsOne = sequencesEncodedAsOne
        self.verbose = verbose
        self.p, self.lr, self.bs = 0.2, 1e-3, 10000 if isHPC else 3500 # 1200 for 2GB VRAM

        self.has_seq_branch, self.has_epi_branch = seqDim > 0, epiDim > 0
        if verbose: print('has_seq_branch:', self.has_seq_branch, 'has_epi_branch:', self.has_epi_branch)

        self.seq_conv1 = Conv1D(256, 5, kernel_initializer='random_uniform', name='seq_conv1', activation=tf.nn.relu)
        self.seq_pool1 = MaxPooling1D(2, name='seq_pool1')
        self.seq_drop1 = keras.layers.Dropout(0.2)
        self.gru1 = Bidirectional(GRU(256, kernel_initializer='he_normal',
                                      dropout=0.3, recurrent_dropout=0.2,
                                      reset_after=False), name='gru1')
        self.seq_dense1 = Dense(256, name='seq_dense1', activation=tf.nn.relu)
        self.seq_drop2 = keras.layers.Dropout(0.3)
        self.seq_dense2 = Dense(128, name='seq_dense2', activation=tf.nn.relu)
        self.seq_drop3 = keras.layers.Dropout(0.2)
        self.seq_dense3 = Dense(64, name='seq_dense3', activation=tf.nn.relu)
        self.seq_drop4 = keras.layers.Dropout(0.2)
        self.seq_dense4 = Dense(40, name='seq_dense4', activation=tf.nn.relu)
        self.seq_drop5 = keras.layers.Dropout(0.2)


        self.epi_conv1 = Conv1D(256, 5, name='epi_conv1', activation=tf.nn.relu)
        self.epi_pool1 = MaxPooling1D(2, name='epi_pool1')
        self.epi_drop1 = keras.layers.Dropout(0.3)
        self.epi_dense1 = Dense(256, name='epi_dense1', activation=tf.nn.relu)
        self.epi_drop2 = keras.layers.Dropout(0.2)
        self.epi_dense2 = Dense(128, name='epi_dense2', activation=tf.nn.relu)
        self.epi_drop3 = keras.layers.Dropout(0.3)
        self.epi_dense3 = Dense(64, name='epi_dense3', activation=tf.nn.relu)
        self.epi_drop4 = keras.layers.Dropout(0.3)
        self.epi_dense4 = Dense(40, name='epi_dense4', activation=tf.nn.relu)


        self.seq_reshape = Reshape((1, 40,), input_shape=(40,))
        self.seq_epi_mult = Multiply()
        self.seq_epi_drop = keras.layers.Dropout(0.2)
        self.seq_epi_flat = Flatten()

        self.seq_epi_output = Dense(1, activation='linear', name='seq_epi_output', input_shape=(-1, 360))

    def call(self, x, training=None):
        if tf.shape(x).shape[0] < 3: x = tf.function(vecToMatEncoding)(x, numBpWise=self.numBpWise, seqDim=self.seqDim, single=self.sequencesEncodedAsOne, setOR=not self.sequencesEncodedAsOne, bs=tf.shape(x)[0]) # decorating with tf.function makes Python functions run faster
        # Calling vecToMatEncoding for each batch is around a factor of 10 slower than calling it outside, but means that we can feed the same encoding into the explainer for each mode. So call it outside if we don't intend to run explainer
        seq_input = tf.transpose(x, perm=[0, 2, 1])  # to match order of dimensions in C-RNNCrispr
        seq_input, epi_input = seq_input[:, :, :self.seqDim], seq_input[:, :, self.seqDim:]

        if self.has_seq_branch:
            seq_x = self.seq_pool1(self.seq_conv1(seq_input))
            seq_x = self.gru1(self.seq_drop1(seq_x, training=training))
            seq_x = self.seq_dense1(seq_x)
            seq_x = self.seq_dense2(self.seq_drop2(seq_x, training=training))
            seq_x = self.seq_dense3(self.seq_drop3(seq_x, training=training))
            seq_x = self.seq_dense4(self.seq_drop4(seq_x, training=training))
            seq_x = self.seq_drop5(seq_x, training=training)

        if self.has_epi_branch:
            epi_x = self.epi_pool1(self.epi_conv1(epi_input))
            epi_x = self.epi_dense1(self.epi_drop1(epi_x, training=training))
            epi_x = self.epi_dense2(self.epi_drop2(epi_x, training=training))
            epi_x = self.epi_dense3(self.epi_drop3(epi_x, training=training))
            epi_x = self.epi_dense4(self.epi_drop4(epi_x, training=training))

        if self.has_seq_branch and self.has_epi_branch:
            seq_x = self.seq_reshape(seq_x)
            comb_x = self.seq_epi_flat(self.seq_epi_drop(self.seq_epi_mult([seq_x, epi_x]), training=training))
        elif self.has_epi_branch:
            comb_x = self.seq_epi_flat(epi_x)
        elif self.has_seq_branch:
            comb_x = self.seq_epi_flat(seq_x)

        comb_x = self.seq_epi_output(comb_x)
        return comb_x

    def reinit_weights(self):
        #initial_weights = self.get_weights()
        #backend_name = K.backend()
        #k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
        #new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]
        #self.set_weights(new_weights)
        self.set_weights(self.initial_weights)

    def set_params(self, p=None, lr=None, bs=None):  # bs is not needed here but included so we can unpack the full hyperparameter array as arguments
        if p is not None:   self.p = p
        if lr is not None: self.lr = lr
        if bs is not None: self.bs = int(bs)

        # set dropout p
        self.seq_drop1 = keras.layers.Dropout(p)
        self.seq_drop3 = keras.layers.Dropout(p)
        self.seq_drop4 = keras.layers.Dropout(p)
        self.seq_drop5 = keras.layers.Dropout(p)
        self.epi_drop2 = keras.layers.Dropout(p)

        self.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics=['mae', 'mse'], run_eagerly=True)

    def get_params(self):
        return self.p, self.lr, self.bs

    def printParams(self):
        print("dropout probability", self.p)

    def load(self, weight_file):
        if weight_file is not None:
            # _ = model(tf.random.normal((2, 23, 4)),tf.random.normal((2, 23, 4))) # to instantiate layers in model
            if weight_file == "data/C_RNNCrispr_weights.h5":
                self.load_from_file(model, seq_branch=self.has_seq_branch,
                               epi_branch=self.has_epi_branch,
                               stem_branch=True)
            else:
                self.load_weights(weight_file)

    # function to load tf v1 weights in C-RNNCrispr paper
    def load_from_file(self, weight_file="data/C_RNNCrispr_weights.h5", seq_branch=True, epi_branch=True, stem_branch=True):
        # adopt a stubborn way to load weights into the neural net
        hf = h5py.File(weight_file, 'r')
        if self.verbose:
            if seq_branch:
                print('load seq_branch weights')
            if epi_branch:
                print('load epi_branch weights')
            if stem_branch:
                print('load stem_branch weights')

        epi_weights = [
            (self.epi_conv1.bias, 'epi_conv1/epi_conv1_292/bias:0'),
            (self.epi_conv1.kernel, 'epi_conv1/epi_conv1_292/kernel:0'),
            (self.epi_dense1.bias, 'epi_dense1/epi_dense1_292/bias:0'),
            (self.epi_dense1.kernel, 'epi_dense1/epi_dense1_292/kernel:0'),
            (self.epi_dense2.bias, 'epi_dense2/epi_dense2_292/bias:0'),
            (self.epi_dense2.kernel, 'epi_dense2/epi_dense2_292/kernel:0'),
            (self.epi_dense3.bias, 'epi_dense3/epi_dense3_292/bias:0'),
            (self.epi_dense3.kernel, 'epi_dense3/epi_dense3_292/kernel:0'),
            (self.epi_dense4.bias, 'epi_dense4/epi_dense4_292/bias:0'),
            (self.epi_dense4.kernel, 'epi_dense4/epi_dense4_292/kernel:0')
        ] if epi_branch else []

        seq_weights = [
            (self.seq_conv1.bias, 'seq_conv1/seq_conv1_292/bias:0'),
            (self.seq_conv1.kernel, 'seq_conv1/seq_conv1_292/kernel:0'),
            (self.seq_dense1.bias, 'seq_dense1/seq_dense1_292/bias:0'),
            (self.seq_dense1.kernel, 'seq_dense1/seq_dense1_292/kernel:0'),
            (self.seq_dense2.bias, 'seq_dense2/seq_dense2_292/bias:0'),
            (self.seq_dense2.kernel, 'seq_dense2/seq_dense2_292/kernel:0'),
            (self.seq_dense3.bias, 'seq_dense3/seq_dense3_292/bias:0'),
            (self.seq_dense3.kernel, 'seq_dense3/seq_dense3_292/kernel:0'),
            (self.seq_dense4.bias, 'seq_dense4/seq_dense4_292/bias:0'),
            (self.seq_dense4.kernel, 'seq_dense4/seq_dense4_292/kernel:0'),
            (self.gru1.backward_layer.cell.kernel, 'gru1/gru1_292/backward_gru_293/kernel:0'),
            (self.gru1.forward_layer.cell.kernel, 'gru1/gru1_292/forward_gru_293/kernel:0'),
            (self.gru1.backward_layer.cell.recurrent_kernel, 'gru1/gru1_292/backward_gru_293/recurrent_kernel:0'),
            (self.gru1.forward_layer.cell.recurrent_kernel, 'gru1/gru1_292/forward_gru_293/recurrent_kernel:0'),
            (self.gru1.backward_layer.cell.bias, 'gru1/gru1_292/backward_gru_293/bias:0'),
            (self.gru1.forward_layer.cell.bias, 'gru1/gru1_292/forward_gru_293/bias:0')] \
            if seq_branch else []

        stem_weights = [(self.seq_epi_output.bias, 'dense_293/dense_293/bias:0'),
                        (self.seq_epi_output.kernel, 'dense_293/dense_293/kernel:0')
                        ] if stem_branch else []

        weights_to_load = seq_weights + epi_weights + stem_weights
        for weight, weight_name in weights_to_load:
            arr = np.array(hf.get(weight_name))
            # print(weight_name, weight.shape, arr.shape)
            weight.assign(arr)


class mySequential(torch.nn.Sequential): # can handle multiple inputs between layers
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple: input = module(*input)
            else:                    input = module(input)
        return input

class vecToMatEncoder(torch.nn.Module):
    def __init__(self, siamese, seqDim, interfaceMode, numBpWise, setOR, CRISPRNetStyle=False):
        super(vecToMatEncoder, self).__init__()
        self.siamese, self.seqDim, self.interfaceMode, self.numBpWise, self.setOR = siamese, seqDim, interfaceMode, numBpWise, setOR
        self.sequencesEncodedAsOne = self.interfaceMode or CRISPRNetStyle

    def forward(self, dataset):
        return vecToMatEncoding(dataset, seqDim=self.seqDim, single=self.sequencesEncodedAsOne, numBpWise=self.numBpWise, setOR=self.setOR)

class ConjoinedConvolutionalNet(torch.nn.Module):
    def __init__(self, regression, p=0.0, device=None, seqDim=4, epiStart=4, epiDim=10, mean = 0, stdev = 0.01, batchnorm_momentum = 0.1):
        super(ConjoinedConvolutionalNet, self).__init__()
        self.device = device
        self.seqDim, self.epiDim, self.epiStart = seqDim, epiDim, epiStart
        self.p, self.mean, self.stdev, self.batchnorm_momentum = p, mean, stdev, batchnorm_momentum
        self.regression = regression
        self.BatchNorm = torch.nn.BatchNorm1d(seqDim+epiDim, momentum=batchnorm_momentum)
        self.conjoinedLayer1 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1), # preserve dimensions
            torch.nn.ReLU()
            )
        self.conjoinedLayer2 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
            torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU()
            )
        self.conjoinedLinear = torch.nn.Linear(512, 1)
        
        self.EncodeLayer2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.2)
            )
        self.EncodeLayer3 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.2)
            )
        self.EncodeLinear = torch.nn.Linear(64*4, 64*4)
        
        self.set_params(p, None, stdev, batchnorm_momentum)

    def reinit_weights(self):
        self.__init__(regression=self.regression, p=self.p, device=self.device, seqDim=self.seqDim, epiStart=self.epiStart, epiDim=self.epiDim, mean=self.mean, stdev=self.stdev, batchnorm_momentum=self.batchnorm_momentum)
        for child in self.modules():
            try:
                torch.nn.init.xavier_uniform_(child.weight)
            except Exception as e:
                pass

    def set_params(self, p=None, lr=None, stdev=None, batchnorm_momentum=None): # lr is not needed here but included so we can unpack the full hyperparameter array as argument
        if p is not None: self.p = p
        if stdev is not None: self.stdev = stdev
        if batchnorm_momentum is not None: self.batchnorm_momentum = batchnorm_momentum

        # set dropout p, Gaussian stdev
        # don't set batchnorm_momentum as this will probably mess up the running stats
        self.EncodeLayer1 = torch.nn.Sequential(
            torch.nn.Conv1d(int(self.seqDim+self.epiDim), 64, kernel_size=3, stride=2, padding=0),
            #nn.BatchNorm1d(32),
            GaussianNoise(self.mean, self.stdev),
            torch.nn.Dropout(p=self.p),
            torch.nn.LeakyReLU(0.2)
            )
        self.EncodeLayer1.to(self.device)
        
        if (self.device != "cpu"):
            for children in self.modules():
                if type(children)==torch.nn.BatchNorm1d:
                    pass#children.track_running_stats = False
        
    def get_params(self):
        return self.p, self.stdev, self.batchnorm_momentum

    def printParams(self):
        print("batchnorm_momentum", self.batchnorm_momentum)
        print("dropout probability", self.p)
        print("Gaussian noise stdev", self.stdev)
    
    def forward_encode(self, x):
        out = self.EncodeLayer1(x)
        out = self.EncodeLayer2(out)
        out = self.EncodeLayer3(out)
        latentRepres = out
        return latentRepres

    def forward(self, x_target, x_grna):
        # train encoder twice: run EncoderConvolutionalNet for x_target, x_grna respectively
        # TODO: import pretrained weights from autoencoder for the two separate networks below
        if (type(x_target) is not torch.Tensor): # arguments can be supplied as lists too
            x_target, x_grna = torch.FloatTensor(x_target).to(self.device), torch.FloatTensor(x_grna).to(self.device)

        #import pandas as pd  # debug
        #x_temp = x_target[0].cpu().numpy()
        #x_temp = pd.DataFrame(x_temp)
        #with pd.option_context('display.max_rows', 1000, 'display.max_columns', None): display(x_temp)  # debug
        #exit()

        x_target, x_grna = x_target.view(x_target.size(0), self.epiStart+self.epiDim, -1).to(self.device), x_grna.view(x_grna.size(0), self.epiStart+self.epiDim, -1).to(self.device)

        x_target, x_grna = self.BatchNorm(x_target), self.BatchNorm(x_grna)
        target_encoding = self.forward_encode(x_target)
        grna_encoding = self.forward_encode(x_grna)
        
        x = torch.cat((target_encoding, grna_encoding), 1) # concatenate model outputs
        out = self.conjoinedLayer1(x)
        out = self.conjoinedLayer2(out)
        out = out.reshape(out.size(0), -1) # flatten last axis to 512x1
        out = self.conjoinedLinear(out)
        if (not self.regression): out = torch.sigmoid(out) # for classification
        return out

def vecToOrEncoding(x, seqDim=4):
    if isinstance(x, tf.Tensor):
        x = tf.map_fn(lambda row: tf.concat([tf.clip_by_value(tf.math.add(row[:23*seqDim], row[23*seqDim:2*23*seqDim]), 0, 1), row[2*23*seqDim:]], axis=0), x)
        return x
    else:
        newshape = list(x.shape)
        newshape[1] -= 23*seqDim
        tmp = torch.empty(newshape)
        for i in range(newshape[0]):
            tmp[i] = torch.cat((torch.clamp(torch.add(x[i][:23*seqDim], x[i][23*seqDim:2*23*seqDim]), 0.0, 1.0), x[i][2*23*seqDim:]))
        return tmp

def vecToMatEncoding(X, binary=False, single=False, seqDim=4, numBpWise=0, setOR=False, bs=None):
    # for every given entry in torch tensor X, transpose the first 2*23 4-entry sets (one-hot sequence encoding), repeat the rest of the vector (epigenetics) below to fill the resulting matrix
    # if numBpWise > 0, transpose the first numBpWise*23 4-entry sets after the sequences as well and append before the epigenetics
    # if X is a tf.Tensor make sure to include the batch size in the arguments
    # TODO: if binary=True, threshold all elements to 0~1

    if bs is None: numDatapoints = list(X.shape)[0]
    else: numDatapoints = bs

    # OR together the second 23*seqDim entries of X to first
    if setOR: # assume guide-target encoding
        X = vecToOrEncoding(X, seqDim)
        single = True # only one sequence part left over

    numSequences = 1 if single else 2
    bpwise_features = []
    if isinstance(X, tf.Tensor): # tf
        for i in range(numSequences):
            bpwise_features.append(tf.transpose(tf.reshape(X[:, i*23*seqDim:(i+1)*23*seqDim], [numDatapoints, 23, seqDim]), perm=[0,2,1]))
        for i in range(numBpWise):
            bpwise_features.append(tf.transpose(tf.reshape(X[:, numSequences*23*seqDim+i*23:numSequences*23*seqDim+(i+1)*23], [numDatapoints, 23, 1]), perm=[0,2,1]))
        sequence_target              = bpwise_features[0] # view target and guide sequences as 2D arrays for each data point
        if not single: sequence_grna = bpwise_features[1]
        epigenetics = tf.keras.backend.repeat_elements(X[:, numSequences*23*seqDim+numBpWise*23:, None], 23, axis=2)  # None indexing means adding an axis - could also use .unsqueeze() here
        # TODO: obtain different epigenetics channel for grna

        # then concatenate epigenetics and bpwise_features onto sequence_target, and sequence_grna if not single
        if single: return tf.concat([sequence_target]+bpwise_features[1:]+[epigenetics], axis=1)
        else:      return tf.concat([sequence_target]+bpwise_features[2:]+[epigenetics], axis=1), tf.concat([sequence_grna]+bpwise_features[2:]+[epigenetics], axis=1)

    else: # torch, xgboost
        for i in range(numSequences):
            bpwise_features.append(X[:, i*23*seqDim:(i+1)*23*seqDim].view(numDatapoints, 23, seqDim).transpose(1,2))
        for i in range(numBpWise):
            bpwise_features.append(X[:, numSequences*23*seqDim+i*23:numSequences*23*seqDim+(i+1)*23].view(numDatapoints, 23, 1).transpose(1,2))
        sequence_target              = bpwise_features[0] # view target and guide sequences as 2D arrays for each data point
        if not single: sequence_grna = bpwise_features[1]
        epigenetics = X[:, numSequences*23*seqDim+numBpWise*23:, None].repeat(1, 1, 23)  # None indexing means adding an axis - could also use .unsqueeze() here
        # TODO: obtain different epigenetics channel for grna

        # then concatenate epigenetics and bpwise_features onto sequence_target, and sequence_grna if not single
        if single: return torch.cat((sequence_target,)+tuple(bpwise_features[1:])+(epigenetics,), dim=1)
        else:      return torch.cat((sequence_target,)+tuple(bpwise_features[2:])+(epigenetics,), dim=1), torch.cat((sequence_grna,)+tuple(bpwise_features[2:])+(epigenetics,), dim=1)


def plotAUC(dataPortions, aucs_roc, aucs_prc, mode, home, dbFile="offtarget_060619.db"):
    # plot AUCs for all the data portions
    plt.plot(dataPortions, aucs_roc, label='ROC')
    plt.plot(dataPortions, aucs_prc, label='PRC')
    # plt.plot(epochs, testlosses, label='final training loss')
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.xlabel('% data')
    plt.ylabel('AUC')

    if (dbFile is not None):
        # find out where new studies start to plot vertical lines
        from setup_db import createConnection, dbQuery
        conn = createConnection(dbFile)
        if conn is not None:
            cursor = dbQuery(conn, "SELECT COUNT(*) FROM cleavage_data GROUP BY experiment_id ORDER BY experiment_id")
            experiments = cursor.fetchall()
            experiments = np.array([experiments[i][0] for i in range(len(experiments))])
            experiments = np.cumsum(experiments / sum(experiments))
            for portion in experiments:
                xcoord = portion  # for a plot from 0 to 1
                plt.axvline(xcoord, color='k', ls='--', lw=1)

    plt.savefig('training_' + mode + '_' + home.split('/')[-1] + '_dataportion.pdf')
    plt.close()

