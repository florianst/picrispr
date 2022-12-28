from models import *
from load_data import inverse_scale_CF
import torch
import torch.utils.data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
from copy import deepcopy
import pandas as pd
import pickle
import re
import numpy as np
from tqdm import tqdm


class CSVDataset():
    def __init__(self, path):
        self.path = path
        self.power_transformers = {}

    def getDataMatrix(self, dbFields, featureEncoding, CFtoScore, chooseSpecies="", booleanX=False, cutoff=1e-5, ext=False, featurenames=[], mode="torch", numBpWise=0, 
                      test_size=0.2, makeSparse=True, doSplit=True, filenameAppendix="", debug_mode=False, CRISPRNetStyle=False, replace_all_nans=False):
        df = pd.read_csv(self.path, low_memory=False)
        if 'experiment_id' not in df.columns: df['experiment_id'] = [0]*df.shape[0]
        if 'genome' not in df.columns: df['genome'] = ['hg19']*df.shape[0]
        if 'cleavage_freq' not in df.columns: 
            self.cf = None
            CFtoScore = lambda x, _: (np.array([0]*len(x)), None) # don't do CF transformation in case it wasn't given in the dataset
            df['cleavage_freq'] = [1]*df.shape[0]
        else:
            self.experiment_ids = df['experiment_id'] # needed for back-transformation of PowerTransformer later, if desired
            self.cf = df['cleavage_freq']
            _, self.power_transformer = CFtoScore(self.cf)

        df.sort_values('experiment_id', inplace=True)
        self.sorted_input_index = df.index
        df.reset_index(inplace=True)

        num_experiments = max(df['experiment_id'])
        
        dbFields = dbFields.split(", ") + ["cleavage_freq"]

        data_measured, data_augmented, weights_measured, weights_augmented = [], [], [], []
        count_experiment, count_augmented = (0,0)

        # normalise CRISPRspec energy values to maximum value
        energy_min, energy_max = -69.823, 38.438451362706445 # gained from full crisprSQL dataset
        norm = energy_max - energy_min

        # replicate datamatrix behaviour: go through energy fields in dbFields, assemble them under their name using normalisation below
        for i, dbField in enumerate(dbFields):
            if "energy" in dbField:
                if (energy_min < 0): dbFieldCalc = re.sub(r'(energy_[0-9]*)', r'(df.\1-'+str(abs(energy_min))+')/'+str(norm), dbField)
                else:                dbFieldCalc = re.sub(r'(energy_[0-9]*)', r'(df.\1+'+str(energy_min)+')/'+str(norm), dbField)

                try:
                    pd.eval("temp_"+str(i)+" = "+dbFieldCalc, target=df, inplace=True) # can't have operators (-) in column name - rename temp column afterwards
                    df.drop(dbField, axis=1, inplace=True, errors='ignore') # drop original dbField if it exists
                    df.rename({"temp_"+str(i): dbField}, axis=1, inplace=True) # rename field to original name
                except AttributeError as e: pass

        def formatDefaults(defaultval):
            if type(defaultval) in [float, np.float64]: return defaultval
            else: return '['+ ' '.join(['{:.3f}'.format(d) for d in defaultval]) + ']'
                
        for dbField in dbFields: # try to find default values for the current model and dbField
            if dbField not in df.columns:
                feature_not_found = True
                if os.path.isfile("default_vals/defaultvals_"+mode+filenameAppendix+".pickle"):
                    defaultvals = pickle.load(open("default_vals/defaultvals_"+mode+filenameAppendix+".pickle", "rb"))
                    if dbField in defaultvals.keys():
                        df[dbField] = [formatDefaults(defaultvals[dbField])]*df.shape[0]
                        print("Used default values for feature", dbField, "- this can decrease prediction accuracy")
                        feature_not_found = False

                if feature_not_found:
                    df[dbField] = [0]*df.shape[0]
                    print("WARNING: Feature", dbField, "was not given and no default values could be found. It will be filled with zeros, which will likely lead to decreased prediction accuracy.")

            num_nans = df[dbField].isna().sum()
            if replace_all_nans and num_nans > 0: # column contains nans - try to fill with default values
                if os.path.isfile("default_vals/defaultvals_"+mode+filenameAppendix+".pickle"):
                    defaultvals = pickle.load(open("default_vals/defaultvals_"+mode+filenameAppendix+".pickle", "rb"))
                    if dbField in defaultvals.keys():
                        df[dbField] = df[dbField].where(~df[dbField].isna(), formatDefaults(defaultvals[dbField]))
                        print("WARNING: Feature", dbField, "has", num_nans, "nan values which were replaced by default values.")
                    else:
                        print("WARNING: Feature", dbField, "has nan values but no default values could be found. Nan values will persist.")
        
        # make sure cleavage_freq is given for all rows
        df['cleavage_freq'].fillna(0, inplace=True)

        # transform cleavage frequencies for each experiment
        for i in range(num_experiments+1):
            mask = df['experiment_id'] == i
            
            mask_species = None
            for species in chooseSpecies:
                if mask_species is None: mask_species = (df['genome'] == species)
                else: mask_species = mask_species | (df['genome'] == species)
            if mask_species is not None: mask = mask & mask_species

            experiment = df[mask & (df['cleavage_freq'] >= cutoff)][dbFields].values.tolist()
            augmented  = df[mask & (df['cleavage_freq'] <  cutoff)][dbFields]
            
            if len(experiment) > 1:
                # set cleavage_freq to -4 for augmented data points
                augmented = augmented.assign(cleavage_freq=-4).values.tolist()
                    
                experiment, augmented, encoding, pt = featureEncoding(experiment, augmented, cutoff, CFtoScore, count=i, i=i+1, featurenames=featurenames, numBpWise=numBpWise, CRISPRNetStyle=CRISPRNetStyle)

                # save PowerTransformer object so we can invert transformation later
                self.power_transformers[i] = pt

                if not doSplit:
                    # assemble experiment and augmented back into the order/indices they had in df
                    # assume that df is passed with non-interspersed experiment_ids
                    indices_experiment = df[mask & (df['cleavage_freq'] >= cutoff)].index.values.tolist()
                    indices_augmented  = df[mask & (df['cleavage_freq'] <  cutoff)].index.values.tolist()
                    start_index = min(indices_experiment + indices_augmented)
                    indices_experiment = [i - start_index for i in indices_experiment]
                    indices_augmented  = [i - start_index for i in indices_augmented]
                    if debug_mode:
                        for i in range(len(experiment) + len(augmented)):
                            if i not in indices_experiment and i not in indices_augmented:
                                print("ERROR: index", i, "not found in experiment nor augmented")

                    assembled = [0] * (len(experiment) + len(augmented))
                    for i, idx in enumerate(indices_experiment):
                        assembled[idx] = experiment[i]
                    for i, idx in enumerate(indices_augmented):
                        assembled[idx] = augmented[i]
                    data_measured.extend(assembled)
                    weights_measured.extend([1]*len(assembled))
                    count_experiment += len(assembled)
                    experiment, assembled, augmented = [], [], []

                else:
                    # add to data set study by study
                    data_measured.extend(experiment)
                    weights_measured.extend([1]*len(experiment))
                    count_experiment += len(experiment)
                    experiment = []
                    data_augmented.extend(augmented)
                    weights_augmented.extend([0]*len(augmented))
                    count_augmented += len(augmented)
                    augmented = []


        weight_augmented = count_experiment / count_augmented if doSplit else 0

        # split up data array: data points X and labels y
        y_measured  = [row.pop(-1) for row in data_measured]
        y_augmented = [row.pop(-1) for row in data_augmented]
        
        if doSplit:
            # separate into training and test data
            from sklearn.model_selection import train_test_split
            import operator
            from itertools import starmap
            # split both measured and augmented points equally over training and test set
            X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(data_measured, y_measured, weights_measured, test_size=test_size, shuffle=True)
            # do this inline to save memory - use the starmap and += iadd operator combination
            X_train, X_test, y_train, y_test, weights_train, weights_test = starmap(operator.iadd, zip((X_train, X_test, y_train, y_test, weights_train, weights_test), train_test_split(data_augmented, y_augmented, weights_augmented, test_size=test_size, shuffle=True)))

            weights = weights_train
            weights.extend(weights_test)

        else:
            # put whole given dataset into xtest/ytest, do not shuffle it - assume all data is in _measured
            X_test, y_test, weights_test = data_measured, y_measured, weights_measured
            X_train, y_train, weights = [[]], [[]], [1]*len(X_test)
        
        # turn into torch tensors - both X and Xtest will be saved as sparse torch tensors
        if booleanX:
            x = torch.as_tensor(X_train.astype(bool))
            xtest = torch.as_tensor(X_test.astype(bool))
        else:
            x = torch.as_tensor(X_train).float()
            xtest = torch.as_tensor(X_test).float()

        del X_train, X_test

        meta_train, meta_test = x[:, :2], xtest[:, :2] # placeholder for meta - is not used with a CSVDataset

        if makeSparse: 
            x = to_sparse(x)
            xtest = to_sparse(xtest)
        
        y = torch.as_tensor(y_train).float()
        ytest = torch.as_tensor(y_test).float()
        
        dm = DataMatrix(mode, verbose=True)
        dm.set_data(x, y, xtest, ytest, meta_train, meta_test, weights, weight_augmented, encoding, featureEncoding.__name__, numBpWise)
        return dm

class FeatureEncoding():
    def __init__(self, epiDim, epiStart, seqDim, featurenames=['A', 'T', 'C', 'G', 'CTCF', 'DNase', 'RRBS', 'H3K4me3', 'DRIP', 'energy1', 'energy2', 'energy3', 'energy4', 'energy5'], encodingFunction="", CRISPRNetStyle=False):
        self.epiDim, self.seqDim, self.epiStart = epiDim, seqDim, epiStart
        self.featureNames = featurenames
        self.interfaceMode = None
        self.encodingFunction = encodingFunction

class DataMatrix():
    def __init__(self, mode="torch", verbose=False):
        self.mode = mode
        # use GPU device if available
        self.check_device(verbose)

    def check_device(self, verbose=False):
        # use GPU device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if (self.device != "cpu") and verbose: print("using GPU device", torch.cuda.get_device_name(torch.cuda.current_device()))
        except AssertionError as e:
            if verbose: print("Torch not compiled with CUDA - using CPU")

    def set_data(self, x, y, xtest, ytest, meta_train, meta_test, weights, weight_augmented, encoding, encodeFunction="", numBpWise=0):
        self.x, self.y, self.xtest, self.ytest, self.meta_train, self.meta_test, self.weights, self.weight_augmented, self.encoding = x, y, xtest, ytest, meta_train, meta_test, weights, weight_augmented, encoding
        self.home = None
        self.weights_ext, self.weight_ext_augmented, self.x_ext, self.y_ext, self.meta_ext = self.weights, self.weight_augmented, self.xtest, self.ytest, self.meta_test # default values for the external test dataset
        self.encodeFunction = encodeFunction
        self.numBpWise = numBpWise
        self.filter = False # whether low-nuc value training points are filtered out
    
    def setExtExperimentIds(self, experiment_ids=[], removeFromTrainSet=True, onlyTestToExt=True): # if onlyTestToExt=False, ext data also contains the test data points of the given experiment IDs
        if (experiment_ids is None or experiment_ids == []): return False

        # extract the passed experiment_ids from current dataframe, save in ext
        dm_temp = self.getConditional(-1, conditions=experiment_ids, include=True,  apply_ext=False)
        if onlyTestToExt: weights_ext, weight_ext_augmented, x_ext, y_ext, meta_ext = dm_temp.weights[list(dm_temp.x.size())[0]:], dm_temp.weight_augmented,                       dm_temp.xtest,                         dm_temp.ytest,                                       dm_temp.meta_test
        else:             weights_ext, weight_ext_augmented, x_ext, y_ext, meta_ext = dm_temp.weights,                             dm_temp.weight_augmented, torch.cat((dm_temp.x, dm_temp.xtest)), torch.cat((dm_temp.y, dm_temp.ytest)), np.concatenate((dm_temp.meta_train, dm_temp.meta_test))

        del dm_temp

        if removeFromTrainSet:
            # take these experiments out of main dataframe
            dm_temp = self.getConditional(-1, conditions=experiment_ids, include=False, apply_ext=False)
            self.x, self.y, self.xtest, self.ytest, self.meta_train, self.meta_test = dm_temp.x, dm_temp.y, dm_temp.xtest, dm_temp.ytest, dm_temp.meta_train, dm_temp.meta_test
            self.weights = dm_temp.weights
            del dm_temp

        # split test in half, one half becomes the new test, other half becomes ext
        if np.sum(np.array(weights_ext) == 1.0) < 15: p = [0.5, 0.5]   # in case there are few measured points, sample equally to increase likelihood of > 2 points in ext set
        else:                                         p = [0.25, 0.75] # empirically optimised ratio
        ind = np.random.choice([True, False], size=(x_ext.shape[0],), p=p) # selection mask between ext (True) and test (False)
        self.weights_ext, self.weight_ext_augmented = weights_ext[ind], weight_ext_augmented
        ind_t = torch.tensor(ind)
        self.x_ext, self.y_ext, self.meta_ext = x_ext[ind_t, :], y_ext[ind_t], meta_ext[ind_t]

        ind = np.logical_not(ind) # invert mask
        self.weights = np.concatenate((self.weights[:list(self.x.size())[0]], weights_ext[ind]))
        ind_t = torch.tensor(ind)
        self.xtest, self.ytest, self.meta_test = x_ext[ind_t, :], y_ext[ind_t], meta_ext[ind_t]

        # rebalance measured/augmented
        self.measuredAugmentedBalance()
        return True

    def measuredAugmentedBalance(self):
        x_size = list(self.x.size())
        weights_train, weights_test = self.weights[:x_size[0]], self.weights[x_size[0]:] # split up combined weights array again (train+test)

        if (sum(weights_train) == len(weights_train)): print("could not adjust weights - all ones or not set")
        elif (self.weight_augmented != 0):
            weight_augmented_train, weight_augmented_test = sum(weights_train)/(len(weights_train) - sum(weights_train)), sum(weights_test)/(len(weights_test) - sum(weights_test))
            self.weight_augmented = np.mean([weight_augmented_train, weight_augmented_test])

        if (sum(self.weights_ext) == len(self.weights_ext)): print("could not adjust ext weights - all ones or not set")
        elif (self.weight_ext_augmented != 0): self.weight_ext_augmented = sum(self.weights_ext)/(len(self.weights_ext) - sum(self.weights_ext))

        return True
        
    def set_extdata(self, x_ext, y_ext):
        self.x_ext, self.y_ext = x_ext, y_ext
        
    def save(self, trainPath, filenameAppendix=''):
        if not os.path.exists(trainPath): os.makedirs(trainPath)
        self.home = trainPath

        if (self.x.type() != 'torch.sparse.FloatTensor'): self.toSparse() # turn X and Xtest into sparse tensors before saving to save filespace
        try:
            pickle.dump(self, open(trainPath+"/dm"+filenameAppendix+".pickle", "wb"), protocol=4) # protocol=4 allows for bigger filesize
        except RuntimeError as e:
            print(e)
        
    def print_information(self, weights_train=None, weights_test=None, weights_ext=None):
        print_weights = ""
        if weights_train is not None:
            unique, counts = np.unique(weights_train, return_counts=True)
            print_weights = ", unique values in weights:"+str(dict(zip(unique, counts)))
        print("training set: x", list(self.x.shape), "y", list(self.y.shape), "of which", (self.y == -4).sum().item(), "entries are -4" + print_weights)
        print_weights = ""

        if weights_test is not None:
            unique, counts = np.unique(weights_test, return_counts=True)
            print_weights = ", unique values in weights:"+str(dict(zip(unique, counts)))
        print("test set: x", list(self.xtest.shape), "y", list(self.ytest.shape), "of which", (self.ytest == -4).sum().item(), "entries are -4" + print_weights)
        print_weights = ""

        if weights_ext is not None:
            unique, counts = np.unique(weights_ext, return_counts=True)
            print_weights = ", unique values in weights:"+str(dict(zip(unique, counts)))
        print("validation set: x", list(self.x_ext.shape), "y", list(self.y_ext.shape), "of which", (self.y_ext == -4).sum().item(), "entries are -4" + print_weights)
    
    def toSparse(self):
        if (self.x.type() != 'torch.sparse.FloatTensor'):     self.x     = self.x.to_sparse()
        if (self.xtest.type() != 'torch.sparse.FloatTensor'): self.xtest = self.xtest.to_sparse()
        if (self.x_ext.type() != 'torch.sparse.FloatTensor'): self.x_ext = self.x_ext.to_sparse()
        
    def toDense(self):
        if (self.x.type() == 'torch.sparse.FloatTensor'):     self.x     = self.x.to_dense()
        if (self.xtest.type() == 'torch.sparse.FloatTensor'): self.xtest = self.xtest.to_dense()
        if (self.x_ext is not None and self.x_ext.type() == 'torch.sparse.FloatTensor'): self.x_ext = self.x_ext.to_dense()
    
    @classmethod              # decorate function such that it becomes a class method and can be called on the class instead of on the instance
    def load(self, trainPath, filenameAppendix=''):
        dm = pickle.load(open(trainPath+"/dm"+filenameAppendix+".pickle", "rb"))
        dm.toDense()
        return dm
    
    def normaliseInput(self, zeroMean=True, unitVariance=False):
        means  = torch.mean(torch.cat((self.x, self.xtest, self.x_ext), dim=0), dim=[0, -1], keepdim=True)
        stdevs = torch.std( torch.cat((self.x, self.xtest, self.x_ext), dim=0), dim=[0, -1], keepdim=True)
        means, stdevs = means.repeat_interleave(list(self.x.size())[-1], dim=-1), stdevs.repeat_interleave(list(self.x.size())[-1], dim=-1)
        
        if zeroMean:
            self.x     = self.x     - means.repeat_interleave(list(self.x.size())[0],     dim=0) 
            self.xtest = self.xtest - means.repeat_interleave(list(self.xtest.size())[0], dim=0) 
            self.x_ext = self.x_ext - means.repeat_interleave(list(self.x_ext.size())[0], dim=0) 
        
        if unitVariance:
            self.x     = self.x     / stdevs.repeat_interleave(list(self.x.size())[0],     dim=0)
            self.xtest = self.xtest / stdevs.repeat_interleave(list(self.xtest.size())[0], dim=0)
            self.x_ext = self.x_ext / stdevs.repeat_interleave(list(self.x_ext.size())[0], dim=0)
            
        return True

    def getConditional(self, column, conditions, include=False, apply_ext=False): # returns new data matrix in which all rows that have the value "condition" at the specified column are removed, or if include=True, only rows that have one or more of the "condition" values at the specified column are kept
        new_dm = deepcopy(self)
        new_dm.toDense()
        
        mask_train = []
        if include:
            # masks contain indices of all elements which are equal to at least one study in conditions, i.e. those we want to keep
            for condition in conditions:
                if len(mask_train) == 0:
                    mask_train = (new_dm.x[:, column]     == condition).nonzero(as_tuple=True)[0]
                    mask_test  = (new_dm.xtest[:, column] == condition).nonzero(as_tuple=True)[0]
                    mask_ext   = (new_dm.x_ext[:, column] == condition).nonzero(as_tuple=True)[0]
                else:
                    mask_train = np.union1d((new_dm.x[:, column]     == condition).nonzero(as_tuple=True)[0], mask_train)
                    mask_test  = np.union1d((new_dm.xtest[:, column] == condition).nonzero(as_tuple=True)[0], mask_test)
                    mask_ext   = np.union1d((new_dm.x_ext[:, column] == condition).nonzero(as_tuple=True)[0], mask_ext)
        else:
            # masks contain indices of all elements which are equal to neither study in conditions, i.e. those we want to keep
            for condition in conditions:
                if len(mask_train) == 0:
                    mask_train = (new_dm.x[:, column]     != condition).nonzero(as_tuple=True)[0]
                    mask_test  = (new_dm.xtest[:, column] != condition).nonzero(as_tuple=True)[0]
                    mask_ext   = (new_dm.x_ext[:, column] != condition).nonzero(as_tuple=True)[0]
                else:
                    mask_train = np.intersect1d((new_dm.x[:, column]     != condition).nonzero(as_tuple=True)[0], mask_train)
                    mask_test  = np.intersect1d((new_dm.xtest[:, column] != condition).nonzero(as_tuple=True)[0], mask_test)
                    mask_ext   = np.intersect1d((new_dm.x_ext[:, column] != condition).nonzero(as_tuple=True)[0], mask_ext)

        new_dm.x = new_dm.x[mask_train]
        new_dm.y = new_dm.y[mask_train]
        mask_train = [int(index) for index in mask_train]
        new_dm.meta_train = np.array(new_dm.meta_train)[mask_train]
        
        new_dm.xtest = new_dm.xtest[mask_test]
        new_dm.ytest = new_dm.ytest[mask_test]
        mask_test = [int(index) for index in mask_test]
        new_dm.meta_test = np.array(new_dm.meta_test)[mask_test]

        if apply_ext: # apply condition also to ext dataset?
            new_dm.x_ext = new_dm.x_ext[mask_ext]
            new_dm.y_ext = new_dm.y_ext[mask_ext]
            mask_ext = [int(index) for index in mask_ext] # need to cast index values to integers
            new_dm.meta_ext = np.array(new_dm.meta_ext)[mask_ext]
            new_dm.weights_ext = np.array(new_dm.weights_ext)[mask_ext]
        
        # adjust weights: split up, apply mask, recombine
        x_size = list(self.x.size())[0]
        mask_train, mask_test = [int(index) for index in mask_train], [int(index) for index in mask_test] # need to cast index values to integers
        new_dm.weights = np.concatenate(((np.array(self.weights)[:x_size])[mask_train], (np.array(self.weights)[x_size:])[mask_test]), axis=None) # can only use indexing on numpy arrays

        return new_dm

    def prepareDataset(self, cutoff_class=-2, addGaussian=False):
        self.toDense()  # training on sparse tensors not implemented yet

        if (not self.regression):  # classification
            # make y and ytest vectors binary (> cutoff_class) for binary classification tasks
            ones = torch.ones(list(self.y.size())[0])
            zeros = torch.zeros(list(self.y.size())[0])
            y = torch.where(self.y > cutoff_class, ones, zeros)

            ones = torch.ones(list(self.ytest.size())[0])
            zeros = torch.zeros(list(self.ytest.size())[0])
            ytest = torch.where(self.ytest > cutoff_class, ones, zeros)

            ones = torch.ones(list(self.y_ext.size())[0])
            zeros = torch.zeros(list(self.y_ext.size())[0])
            y_ext = torch.where(self.y_ext > cutoff_class, ones, zeros)

        else:
            y, ytest, y_ext = self.y, self.ytest, self.y_ext
            if addGaussian: # add a small Gaussian to the lowest values in y to make the label distribution more natural
                #y     += np.abs(np.random.normal(scale=0.3, size=y.shape))     * (y     == -4).detach().cpu().numpy()
                #ytest += np.abs(np.random.normal(scale=0.3, size=ytest.shape)) * (ytest == -4).detach().cpu().numpy()
                y_ext += np.abs(np.random.normal(scale=0.2, size=y_ext.shape)) * (y_ext == -4).detach().cpu().numpy()
                y, ytest, y_ext = y.float(), ytest.float(), y_ext.float()

        self.meta_train, self.meta_test, self.meta_ext = np.array(self.meta_train), np.array(self.meta_test), np.array(self.meta_ext)
        # convert string columns in meta arrays to float so they can become torch tensors
        from sklearn import preprocessing
        label_encoder = preprocessing.LabelEncoder()
        bpwise_columns, label_encodings = [], {}
        for col in range(len(self.meta_train[0])):
            if type(self.meta_train[0][col]) in [str, np.str_] or self.meta_train[0][col] is None: # all non-numerical columns need to go through splitBpwise or labelencoder
                if any([self.meta_train[row][col] is not None and self.meta_train[row][col].find('[') > -1 for row in range(len(self.meta_train))]): # search through whole column
                    self.meta_train[:, col] = [np.mean(splitBpwise(field)) for field in self.meta_train[:, col]]
                    self.meta_test[:, col]  = [np.mean(splitBpwise(field)) for field in self.meta_test[:, col]]
                    self.meta_ext[:, col]   = [np.mean(splitBpwise(field)) for field in self.meta_ext[:, col]]
                else: # torch tensors can only have numerical elements, so need to transform string columns to numerical first
                    # labelencoder does not work for missing values
                    self.meta_train[:, col][pd.isnull(self.meta_train[:, col])] = 'nan' # pd.isnull can handle numpy arrays of dtype object too
                    self.meta_test[:, col][pd.isnull(self.meta_test[:, col])]   = 'nan'
                    self.meta_ext[:, col][pd.isnull(self.meta_ext[:, col])]     = 'nan'

                    self.meta_train[:, col] = label_encoder.fit_transform(self.meta_train[:, col].astype(str))
                    self.meta_test[:, col]  = label_encoder.fit_transform(self.meta_test[:, col].astype(str))
                    self.meta_ext[:, col]   = label_encoder.fit_transform(self.meta_ext[:, col].astype(str))
                    label_encodings[col] = list(label_encoder.classes_)

        try:
            trainDataset = torch.utils.data.TensorDataset(self.x,     y,     torch.as_tensor(self.meta_train.astype(np.float32))) # training dataset (training algorithm optimises this)
            validDataset = torch.utils.data.TensorDataset(self.xtest, ytest, torch.as_tensor(self.meta_test.astype(np.float32)))  # validation dataset (hyperparameter optimisation optimises this)
            extDataset   = torch.utils.data.TensorDataset(self.x_ext, y_ext, torch.as_tensor(self.meta_ext.astype(np.float32)))   # external test dataset (used for ROC, PRC etc.)
        except ValueError as e:
            print(e)
            for col in range(len(self.meta_train[0])): print(type(self.meta_train[0][col]))
            print(self.meta_train)

        return trainDataset, validDataset, extDataset

    def prepareDataloaders(self, trainDataset, validDataset, extDataset, bs=None, exp_ids=((), ()), dataPortion=1.0, balanceClasses=False, verbose=False, ignoreExtSet=False, doSampling=True):
        x_size = list(self.x.size())
        weights_train, weights_test = self.weights[:x_size[0]], self.weights[x_size[0]:]  # split up combined weights array again (train+test)

        weights_train = [self.weight_augmented if weight == 0 else weight for weight in weights_train]
        weights_test  = [self.weight_augmented if weight == 0 else weight for weight in weights_test]
        weights_ext   = [self.weight_ext_augmented if weight == 0 else weight for weight in self.weights_ext]
        if (not self.regression and balanceClasses):  # make sure that class labels are balanced
            # weights_train = classBalance(weights_train, y)
            # weights_test  = classBalance(weights_test,  ytest)
            weights_ext = classBalance(weights_ext, extDataset.tensors[1]) # extDataset.tensors[1] = y_ext (see above)

        # set appropriate weight values to 0
        exp_ids_train, exp_ids_test = exp_ids
        weights_train = self.portionWeights(exp_ids_train, weights_train, dataPortion, list(self.x.size()))
        weights_test  = self.portionWeights(exp_ids_test,  weights_test,  dataPortion, list(self.xtest.size()))

        # adjust relative weight of measured/augmented to new size of dataset (assume this is equal for train and test due to split_train_test)
        if (len(weights_train) != weights_train.count(1) + weights_train.count(0)):  # only necessary if dataset contains augmented points
            weight_augmented_new = weights_train.count(1) / (len(weights_train) - weights_train.count(1) - weights_train.count(0))
            weights_train = [x if (x == 1 or x == 0) else weight_augmented_new for x in weights_train]
            weights_test  = [x if (x == 1 or x == 0) else weight_augmented_new for x in weights_test]

        if verbose: self.print_information(weights_train, weights_test, weights_ext)

        bs_ext = int(7e4) if self.mode != "torch" else int(bs) # training is so far designed to use only one batch per epoch to save memory
        trainSampler = torch.utils.data.WeightedRandomSampler(weights_train, min(int(bs), 60000), replacement=True)  # sample such that on average we get equal numbers of measured and augmented data points
        testSampler  = torch.utils.data.WeightedRandomSampler(weights_test,  min(int(bs), 60000), replacement=True)
        extSampler   = torch.utils.data.WeightedRandomSampler(weights_ext,   min(bs_ext,  60000), replacement=True)

        self.trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=min(int(bs), 60000), num_workers=0, sampler=trainSampler if doSampling else None)
        self.testLoader  = torch.utils.data.DataLoader(dataset=validDataset, batch_size=min(int(bs), 60000), num_workers=0, sampler=testSampler  if doSampling else None)
        if (self.x_ext is not None and not ignoreExtSet):
            # construct an ext dataset that contains all the positives and the appropriate number of randomly sampled negatives
            weights_ext_np = np.array(weights_ext) # go numpy for boolean indexing
            indices_measured = np.arange(len(extDataset))[weights_ext_np == 1.0]
            indices_augmented= np.arange(len(extDataset))[weights_ext_np != 1.0]

            indices_augmented = np.random.choice(indices_augmented, size=int(len(indices_augmented)*self.weight_ext_augmented), replace=True)
            bs_ext = len(indices_measured) + len(indices_augmented)

            self.extLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.Subset(extDataset, np.concatenate((indices_measured, indices_augmented))), batch_size=bs_ext, num_workers=0)  # use dataloader here so passing to function does not create another copy of extDataset
        else:
            # ignore ext set, i.e. choose test as validation set (e.g. in a hyperopt run)
            self.extLoader = self.testLoader  # fallback in case no extDataset was initialised using getDataMatrix(ext=True)
            weights_ext = weights_test

        return weights_train, weights_test, weights_ext

    def portionWeights(self, exp_ids, weights, dataPortion, x_size, verbose=False): # set appropriate weights values to 0
        already_deleted = 0
        left_to_delete = abs(int(x_size[0]*(1-dataPortion)))
        if left_to_delete > 0:
            for exp_id, startindex, length in exp_ids:
                startindex, length = int(startindex), int(length)
                if (left_to_delete >= length): # set whole studies' weights to 0
                    already_deleted += length
                    left_to_delete  -= length
                    weights[startindex:startindex+length] = [0]*length
                    if verbose: print("set complete experiment_id", exp_id, "of length", length, "to zero, left to delete:", left_to_delete)
                else:
                    # get current study's measured/augmented ratio
                    weights = np.array(weights)
                    length_left_measured  = max([0, left_to_delete-len(np.nonzero(np.abs(weights[startindex:startindex+length] - self.weight_augmented) < 1e-4)[0])]) # keep measured points in the dataset for longest (set augmented to zero first)
                    length_left_augmented = min([left_to_delete,   len(np.nonzero(np.abs(weights[startindex:startindex+length] - self.weight_augmented) < 1e-4)[0])])
                    if verbose: print(np.unique(weights[startindex:startindex+length], return_counts=True))
                    indices_augmented = startindex+np.nonzero(weights[startindex:startindex+length] == self.weight_augmented)[0][:length_left_augmented]
                    indices_measured  = startindex+np.nonzero(weights[startindex:startindex+length] == 1                    )[0][:length_left_measured]
                    weights[np.concatenate((indices_augmented, indices_measured), axis=None)] = 0
                    weights = weights.tolist()
                    if verbose: print("set", len(indices_augmented), "indices in augmented and", len(indices_measured), "indices in measured to zero")
                    break
        return weights

    def dropColumn(self, column): # delete specific column from all x arrays in the dataframe
        new_dm = deepcopy(self)

        if new_dm.x.shape[1] > 0: new_dm.x = np.delete(new_dm.x, column, axis=1)
        new_dm.xtest = np.delete(new_dm.xtest, column, axis=1)
        new_dm.x_ext = np.delete(new_dm.x_ext, column, axis=1)

        new_dm.encoding.epiDim -= 1 # assume that column is from the epigenetics part
        new_dm.encoding.featureNames = np.delete(new_dm.encoding.featureNames, column)
        return new_dm
    
class TrainResult():
    def __init__(self, mode, regression, home, filenameAppendix, model, lr, dataPortions=None, final_testloss=None, tpr=[], fpr=[], precision=[], recall=[], aucs_roc=None, aucs_roc_std=None, aucs_prc=None, aucs_prc_std=None, 
                spearmanrs=None, spearmanr_stds=None, pearsonrs=None, pearsonr_stds=None, train_set=None, explain_set=None, meta_train=None, meta_explain=None, encoding=None, siamese=None, interfaceMode=False, exp_ids=None, mismatchType=False, 
                numBpWise=0, indexes_train=[], indexes_test=[], indexes_ext=[], CRISPRNetStyle=False):
        self.mode, self.regression = mode, regression
        self.model = model
        self.lr = lr
        self.home, self.filenameAppendix = home, filenameAppendix
        self.final_testloss, self.tpr, self.fpr, self.precision, self.recall, self.aucs_roc, self.aucs_prc, self.spearmanrs, self.pearsonrs = final_testloss, tpr, fpr, precision, recall, aucs_roc, aucs_prc, spearmanrs, pearsonrs
        self.aucs_roc_std, self.aucs_prc_std, self.spearmanr_stds, self.pearsonr_stds = aucs_roc_std, aucs_prc_std, spearmanr_stds, pearsonr_stds
        self.dataPortions = dataPortions
        self.encoding, self.siamese, self.interfaceMode, self.mismatchType, self.CRISPRNetStyle = encoding, siamese, interfaceMode, mismatchType, CRISPRNetStyle
        self.train_set, self.explain_set = train_set, explain_set
        self.meta_train, self.meta_explain = meta_train, meta_explain
        self.indexes_train, self.indexes_test, self.indexes_ext = indexes_train, indexes_test, indexes_ext
        self.exp_ids = exp_ids
        self.numBpWise = numBpWise
            
    @classmethod
    def load(self, trainPath, mode="torch", filenameAppendix="", resultAppend="", device="cpu"):
        trainResult = pickle.load(open(trainPath+"/"+resultAppend+"trainresult_"+mode+filenameAppendix+".pickle", "rb"))
        if mode == "tf":
            trainResult.model = CRNNCrisprModel(trainResult.regression, device, trainResult.encoding.seqDim, trainResult.encoding.epiStart, trainResult.encoding.epiDim, trainResult.numBpWise, sequencesEncodedAsOne=trainResult.CRISPRNetStyle, dynamic=True)
            weights_pickled = pickle.load(open(trainPath+"/"+resultAppend+"trainresult_"+mode+filenameAppendix+"_weights.pickle", "rb"))

            # compile model
            trainResult.model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'], run_eagerly=True)
            # need to run a subclassed model once to set input/output shapes in order to be able to load weights
            _ = trainResult.model(tf.convert_to_tensor(trainResult.train_set.cpu().detach().numpy()))

            # set weights in model
            names_pickled = [w.name for w in weights_pickled]
            for layer in trainResult.model.layers:
                weights_list = []
                for i, name in enumerate(names_pickled):
                    if layer.name == name.split('/')[1]:
                        weights_list.append(weights_pickled[i].numpy())
                layer.set_weights(weights_list)

        return trainResult


def to_sparse(x):
    """ converts dense tensor x to sparse format
    from https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/2 """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def splitBpwise(bpwise): # split up string representation of 23bp basepair-wise features in the database, return as list of floats
    if bpwise is None: return np.nan
    else:
        bpwise = bpwise.replace("\n", ' ').replace("[", "").replace("]", "").replace("   ", " ").replace("  ", " ").split(" ")
        filtering = filter(lambda x: x not in ['', None], bpwise)  # remove empty elements
        bpwise = [float(i) for i in filtering]
        return bpwise

def classBalance(weights, y): # balance weights array according to binary labels in y
    y_active = np.array(y)[np.array(weights) != 0]
    weight_false = len(np.nonzero(y_active)[0]) / (len(y_active)-len(np.nonzero(y_active)[0]))
    for i in range(len(weights)):
        if weights[i] != 0: weights[i] = 1 if y[i] == 1 else weight_false

    return weights

if __name__ == "__main__":
    import sys
    from encoding import oneHotSingleNuclTargetMismatchType, oneHotSingleNuclTargetMismatch, oneHotSingleNucl, normaliseCF
    from models import mySequential, vecToMatEncoder, vecToMatEncoding
    import pickle
    from tqdm import tqdm
    import xgboost as xgb
    import torch
    import tensorflow as tf
    from scipy.stats import spearmanr
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    use_gpu = True

    filenames = ["xgboost_interface_type_epi", "torch_eng", "torch_engnuc", "torch_interface_type_nuc", "tf_eng", "tf_engnuc"]
    modelnames =["XGB_S3E2", "CNN_S2E0", "CNN_S4E0", "CNN_S5E2", "RNN_S2E3", "RNN_S4E3"]

    # define models
    seq_features = "target_sequence, grna_target_sequence, "
    energy_features = "energy_2+energy_1-(energy_3*energy_4/energy_2), "
    epigen_features = "epigen_ctcf, epigen_dnase, epigen_rrbs, epigen_h3k4me3, epigen_drip, "
    nuc_features = "NucleotideBDM, NuPoP_Affinity_147_human, GCContent, "
    numBpWise = 3

    seq_energy_feat         = seq_features + energy_features + "experiment_id"
    seq_energy_epi_feat     = seq_features + energy_features + epigen_features + "experiment_id"
    seq_energy_nuc_feat     = seq_features + nuc_features + energy_features + "experiment_id"
    seq_energy_epi_nuc_feat = seq_features + nuc_features + energy_features + epigen_features + "experiment_id"

    # retrieve user input
    filePath   = "test_input.csv" if len(sys.argv) <= 1  else sys.argv[1]
    modelNum   = 5                if len(sys.argv) <= 2  else int(sys.argv[2])
    home       = "models"         if len(sys.argv) <= 3  else sys.argv[3]
    regression = False            if len(sys.argv) <= 4  else eval(sys.argv[4])
    replace_all_nans = False      if len(sys.argv) <= 5  else eval(sys.argv[5])

    args = [("", home, "torch", False, regression, "s4", False), # E3+CRISPRNetStyle, CNN, s4
            ("", home, "torch", False, regression, "s2", False), # E3+CRISPRNetStyle, CNN, s2
            ("", home, "tf",    False, regression, "s4", False), # E3+CRISPRNetStyle, RNN, s4
            ("", home, "tf",    False, regression, "s2", False), # E3+CRISPRNetStyle, RNN, s2
            ("", home, "torch", True,  regression, "s4", True),  # E2,                CNN, s4
            ("", home, "torch", True,  regression, "s2", True),  # E2,                CNN, s2
            ("", home, "tf",    True,  regression, "s4", True),  # E2,                RNN, s4
            ("", home, "tf",    True,  regression, "s2", True),  # E2,                RNN, s2
            ]

    kwargs = [{'dbFields': seq_energy_nuc_feat, 'numBpWise': numBpWise, 'CRISPRNetStyle': True},
              {'dbFields': seq_energy_feat,     'numBpWise': 0,         'CRISPRNetStyle': True},
              {'dbFields': seq_energy_nuc_feat, 'numBpWise': numBpWise, 'CRISPRNetStyle': True},
              {'dbFields': seq_energy_feat,     'numBpWise': 0,         'CRISPRNetStyle': True},
              {'dbFields': seq_energy_nuc_feat, 'numBpWise': numBpWise, 'CRISPRNetStyle': False},
              {'dbFields': seq_energy_feat,     'numBpWise': 0,         'CRISPRNetStyle': False},
              {'dbFields': seq_energy_nuc_feat, 'numBpWise': numBpWise, 'CRISPRNetStyle': False},
              {'dbFields': seq_energy_feat,     'numBpWise': 0,         'CRISPRNetStyle': False}
              ]

    # load user dataset and chosen model
    dataset = CSVDataset(filePath)

    config = {"dbFields": kwargs[modelNum]['dbFields'],
              "numBpWise": kwargs[modelNum]['numBpWise'],
              "mode": args[modelNum][2],
              "mismatchType": args[modelNum][3],
              "interfaceMode": args[modelNum][6],
              "chooseSpecies": ["hg19", "hg38"],
              "regression": args[modelNum][4],
              "CRISPRNetStyle": kwargs[modelNum]['CRISPRNetStyle'],
             }

    filenameAppendix  = "_interface" if config["interfaceMode"] else ""
    filenameAppendix += "_type"      if config["mismatchType"] and config["interfaceMode"]  else ""
    filenameAppendix += "_"+args[modelNum][5]
    
    print(config["mode"], filenameAppendix, "__________________________________")

    if config["mismatchType"]:
        oneHotFct = oneHotSingleNuclTargetMismatchType
        featurenames = ['A_match', 'T_match', 'C_match', 'G_match', 
                        'A_mismT', 'T_mismC', 'C_mismG', 'G_mismA', 
                        'A_mismC', 'T_mismG', 'C_mismA', 'G_mismT',
                        'A_mismG', 'T_mismA', 'C_mismT', 'G_mismC']
    elif config["interfaceMode"]: 
        oneHotFct = oneHotSingleNuclTargetMismatch
        featurenames = ['A', 'A_mism', 'T', 'T_mism', 'C', 'C_mism', 'G', 'G_mism']
    else:
        oneHotFct = oneHotSingleNucl
        featurenames = ['A',           'T',           'C',           'G']

    featurenames.extend(" ".join(config["dbFields"].split()).split(', ')[2:]) # append whatever database fields apart from guide and target sequence are used
    
    
    # get data matrix from user input
    filenameAppendix += "_class" if not config["regression"] else ""

    print("encoding dataset...", end = '')
    dm = dataset.getDataMatrix(config["dbFields"], oneHotFct, normaliseCF, chooseSpecies=config["chooseSpecies"], filenameAppendix=filenameAppendix,
                                featurenames=featurenames, mode=config["mode"], numBpWise=config["numBpWise"], test_size=0.2, doSplit=False, 
                                CRISPRNetStyle=config["CRISPRNetStyle"], replace_all_nans=replace_all_nans)
    print("done")

    dm.mode = config["mode"]
    dm.interfaceMode = config["interfaceMode"]
    dm.mismatchType = config["mismatchType"]
    dm.regression = config["regression"]
    dm.CRISPRNetStyle = config["CRISPRNetStyle"]
    
    # load model
    print("loading model...", end = '')
    result = TrainResult.load(home, config["mode"], filenameAppendix, device="gpu:0" if use_gpu else "cpu")
    print("done")
    
    # predict on ext set
    isHPC = True
    bs = int(7e4) if isHPC or dm.mode != "torch" else 35000
    
    # don't use experiment_id column
    dm.toDense()
    dm = dm.dropColumn(-1)
    
    print("preparing dataset...", end = '')
    trainDataset, validDataset, extDataset = dm.prepareDataset(cutoff_class=-4, addGaussian=False)
    dm.prepareDataloaders(trainDataset, validDataset, extDataset, bs, balanceClasses=False, ignoreExtSet=True, doSampling=False)
    print("done")

    if dm.mode == "torch": 
        model = result.model
        if not use_gpu: model.device = "cpu"
        ORencoding = dm.mismatchType and (not dm.interfaceMode) and (not dm.CRISPRNetStyle)
        siamese = (dm.mode == "torch" and not config["interfaceMode"] and not ORencoding and not dm.CRISPRNetStyle)

        model = model.to(model.device)
        model.eval()

    print("obtaining predictions...")

    preds_set, y_ext_set = [], []
    for (x_ext, y_ext, _) in tqdm(dm.testLoader) if len(dm.testLoader) > 2 else dm.testLoader:
        if (dm.mode == "torch"):
            x_ext = vecToMatEncoding(x_ext, seqDim=dm.encoding.seqDim, single=dm.interfaceMode or dm.CRISPRNetStyle, 
                                    numBpWise=dm.numBpWise, setOR=dm.mismatchType and (not dm.interfaceMode) and (not dm.CRISPRNetStyle))
            if siamese:
                x_ext = list(x_ext)
                for i in range(len(x_ext)):
                    x_ext[i] = x_ext[i].to(model.device)
                ypred = model(*x_ext).flatten()
            else:
                x_ext = x_ext.to(model.device)
                ypred = model(x_ext).flatten()
            y_ext, preds = y_ext.detach().cpu().data, ypred.detach().cpu().data.numpy()

        elif (dm.mode == "xgboost"):
            if result.mismatchType and not result.interfaceMode: x_ext = vecToOrEncoding(x_ext, result.encoding.seqDim) # OR encoding
            matrix_ext = xgb.DMatrix(x_ext.cpu().detach().numpy())
            y_ext = y_ext.detach().cpu().data

            preds = result.model.predict(matrix_ext).flatten()

        elif (dm.mode == "tf"):
            preds = result.model.predict(x_ext.numpy())

        preds_set.extend(list(preds.flatten()))
        y_ext_set.extend(y_ext.detach().cpu().data.tolist())

    df = pd.DataFrame()
    df['piCRISPR prediction'] = preds_set
    if dataset.cf is not None: df['ground truth_transformed'] = y_ext_set
    # restore input order
    df.index = dataset.sorted_input_index
    df.sort_index(inplace=True)

    if dataset.cf is not None:
        df['ground truth'] = dataset.cf
        # apply back-transformation of PowerTransformer
        df['piCRISPR prediction_backtransformed'] = df['piCRISPR prediction'].apply(inverse_scale_CF)
        df['piCRISPR prediction_backtransformed'] = dataset.power_transformer.inverse_transform(df['piCRISPR prediction_backtransformed'].to_numpy().reshape(-1, 1)).flatten()
        # normalise
        df['piCRISPR prediction_backtransformed'] = (df['piCRISPR prediction_backtransformed']-df['piCRISPR prediction_backtransformed'].min())/(df['piCRISPR prediction_backtransformed'].max()-df['piCRISPR prediction_backtransformed'].min())
        

        # show some benchmarks
        if regression: 
            print("")
            print("whole dataset, transformed back to [0, 1] domain: Spearman r =", spearmanr(df['ground truth'].fillna(0), df['piCRISPR prediction_backtransformed'].fillna(0))[0])
            print("whole dataset, at transformed [-4, 4] domain: Spearman r =",   spearmanr(df['ground truth_transformed'], df['piCRISPR prediction'])[0])
            print("")

        # save some plots for debugging
        if True:
            plt.scatter(df['ground truth'], df['piCRISPR prediction_backtransformed'], marker='.', alpha=0.2)
            plt.savefig("transformedback.png")
            plt.close()

            plt.scatter(df['ground truth_transformed'], df['piCRISPR prediction'], marker='.', alpha=0.2)
            plt.savefig("nontransformed.png")
            plt.close()

        df['piCRISPR prediction'] = df['piCRISPR prediction_backtransformed'] # only show backtransformed value (in [0, 1] domain)
        df.drop('piCRISPR prediction_backtransformed', axis=1, inplace=True)

    df.to_csv("output.csv", index=False)
    print("successfully saved predictions to output.csv")