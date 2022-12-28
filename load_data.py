import numpy as np
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

def enforceSeqLength(sequence, requireLength):
    if (len(sequence) < requireLength): sequence = '0'*(requireLength-len(sequence))+sequence # in case sequence is too short, fill in zeros from the beginning (or sth arbitrary thats not ATCG)
    return sequence[-requireLength:] # in case sequence is too long

def seqToOneHot(sequence, requireLength, seq_guide=None, mismatch_type=False, CRISPRNetStyle=False):
    sequence = enforceSeqLength(sequence, requireLength)
    bases = ['A', 'T', 'C', 'G']
    
    if (seq_guide is None):
        onehot = np.zeros(4*len(sequence), dtype=int)
        for i in range(len(sequence)):
            for key,base in enumerate(bases):
                if (sequence[i] == base):
                    onehot[4*i+key] = 1
    else:
        if mismatch_type:
            onehot = np.zeros(16*len(sequence), dtype=int)
            for i in range(min(len(sequence), len(seq_guide))):
                for key,base in enumerate(bases):
                    if (sequence[i] == base):
                        for baseoffset in range(len(bases)):
                            if sequence[i] == bases[bases.index(seq_guide[i])-baseoffset]: mismtype = baseoffset
                        onehot[16*i+4*mismtype+key] = 1
        elif CRISPRNetStyle:
            onehot = np.zeros(6*len(sequence), dtype=int)
            for i in range(min(len(sequence), len(seq_guide))):
                for key,base in enumerate(bases):
                    if (sequence[i] == base):
                        onehot[6*i+key] = 1
                    if (seq_guide[i] == base): # OR encoding within CRISPRNetStyle
                        onehot[6*i+key] = 1
                if sequence[i] != seq_guide[i]: # mismatch
                    try:
                        if bases.index(sequence[i]) < bases.index(seq_guide[i]): onehot[6*i+4] = 1
                        else: onehot[6*i+5] = 1
                    except ValueError: # non-ATCG base found
                        pass
        else:
            onehot = np.zeros(8*len(sequence), dtype=int)
            for i in range(min(len(sequence), len(seq_guide))):
                for key,base in enumerate(bases):
                    if (sequence[i] == base):
                        mism = 1 if sequence[i] != seq_guide[i] else 0
                        onehot[8*i+2*key+mism] = 1
        
    return onehot.tolist()

def seqToSW(sequence, requireLength, seq_guide=None): # G, C --> S / A, T --> W one hot
    sequence = enforceSeqLength(sequence, requireLength)
    sw = np.zeros(len(sequence), dtype=int)
    for i in range(len(sequence)):
        if (sequence[i] in ['G', 'C']):
            sw[i] = 0
        else:
            sw[i] = 1
    return sw.tolist()

def seqToRY(sequence, requireLength, seq_guide=None): # A, G --> R / C, T --> Y one hot
    sequence = enforceSeqLength(sequence, requireLength)
    ry = np.zeros(len(sequence), dtype=int)
    for i in range(len(sequence)):
        if (sequence[i] in ['A', 'G']):
            ry[i] = 0
        else:
            ry[i] = 1
    return ry.tolist()

def seqToDinucleotides(sequence, requireLength, seq_guide=None): # 16 channels for the n-1 bases for adjacent dinucleotides
    sequence = enforceSeqLength(sequence, requireLength)

    if (seq_guide is None):
        dinucl = np.zeros(16*(len(sequence)), dtype=int) # only have 23-1 adjacent pairs, but as of now, 23 is hard-coded into train.py and plot_shap.py so leave the last pair to be only zeros
        for i in range(len(sequence)-1):
            for key,bases in enumerate(['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG', 'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG']):
                if (sequence[i:i+2] == bases): # need i+2 here since the last index of the interval is excluded!
                    dinucl[16*i+key] = 1
                    
    else:
        dinucl = np.zeros(32*(len(sequence)), dtype=int) # only have 23-1 adjacent pairs, but as of now, 23 is hard-coded into train.py and plot_shap.py so leave the last pair to be only zeros
        for i in range(len(sequence)-1):
            for key,bases in enumerate(['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG', 'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG']):
                if (sequence[i:i+2] == bases): # need i+2 here since the last index of the interval is excluded!
                    mism = 1 if sequence[i:i+2] != seq_guide[i:i+2] else 0
                    dinucl[32*i+2*key+mism] = 1

    return dinucl.tolist()

def turnToOneHot(experiment, howMany=2, seqToOneHot=seqToOneHot, encodingFunction="", numBpWise=0, CRISPRNetStyle=False): # takes the first two elements in each row of experiment and turns them into one-hot vectors sequentially
    requireLength = 23
    # numpy tolist() doesn't preserve types! i.e. transfers all numbers to strings in the experiment array because of the string sequences
    
    if (encodingFunction in ["oneHotSingleNuclTargetMismatch", "oneHotDinuclTargetMismatch", "oneHotSingleNuclTargetMismatchType", "oneHotDinuclTargetMismatchType"]) or (encodingFunction == "oneHotSingleNucl" and CRISPRNetStyle): # target-mismatch or target-mismatch type encodings
        for key, (sequence_target, sequence_guide) in tqdm(enumerate(zip([experiment[i][0] for i in range(len(experiment))], [experiment[i][1] for i in range(len(experiment))]))):
            onehot = seqToOneHot(sequence_target, requireLength, sequence_guide, encodingFunction in ["oneHotSingleNuclTargetMismatchType", "oneHotDinuclTargetMismatchType"], CRISPRNetStyle=CRISPRNetStyle)
            seqLength = len(onehot)
            # split up string representation of base pair-wise features (come straight after target and guide sequence)
            bpwiseFeatures = experiment[key][2:2+numBpWise]
            bpwiseList = []
            for bpwise in bpwiseFeatures:
                if bpwise is not None and type(bpwise) == str:
                    bpwise = bpwise.replace(r"\n", "").replace("[", "").replace("]", "").replace("   ", " ").replace("  ", " ").split(" ")
                    filtering = filter(lambda x: x not in ['', None], bpwise)  # remove empty elements
                    bpwise = [float(i) for i in filtering]
                    if len(bpwise) == 23: bpwiseList.append(bpwise)
                    else:                 bpwiseList.append([0]*23) # in case nucleosome data is not available
                else: bpwiseList.append([0]*23)
            bpwiseList = [item for sublist in bpwiseList for item in sublist]

            experiment[key] = experiment[key][:2]+onehot+bpwiseList+[float(i) if i is not None else 0 for i in experiment[key][2+numBpWise:]] # insert onehot vector before element 2
            del experiment[key][0] # delete target_sequence of each data point
            del experiment[key][0] # delete guide_sequence  of each data point
            
    else: # target-guide encoding
        for key, sequence in tqdm(enumerate([experiment[i][0] for i in range(len(experiment))])):
            onehot = seqToOneHot(sequence, requireLength)
            seqLength = len(onehot)
            # split up string representation of base pair-wise features (come straight after target and guide sequence)
            bpwiseFeatures = experiment[key][2:2 + numBpWise]
            bpwiseList = []
            for bpwise in bpwiseFeatures:
                if bpwise is not None and type(bpwise) == str:
                    bpwise = bpwise.replace(r"\n", "").replace("[", "").replace("]", "").replace("   ", " ").replace("  ", " ").split(" ")
                    filtering = filter(lambda x: x not in ['', None], bpwise)  # remove empty elements
                    bpwise = [float(i) for i in filtering]
                    if len(bpwise) == 23: bpwiseList.append(bpwise)
                    else:                 bpwiseList.append([0]*23)  # in case nucleosome data is not available
                else: bpwiseList.append([0]*23)
            bpwiseList = [item for sublist in bpwiseList for item in sublist]

            experiment[key] = experiment[key][:2]+onehot+bpwiseList+[float(i) if i is not None else 0 for i in experiment[key][2+numBpWise:]] # insert onehot vector before element 2
            del experiment[key][0] # delete target_sequence of each data point
    
        if (howMany >= 2):
            for key, sequence in enumerate([experiment[i][0] for i in range(len(experiment))]):  # now grna_target_sequence is at index 0
                onehot = seqToOneHot(sequence, requireLength)
                experiment[key] = experiment[key][:seqLength+1]+onehot+[float(i) if i is not None else 0 for i in experiment[key][seqLength+1:]] # insert onehot vector before element 2
                del experiment[key][0] # delete grna_target_sequence of each data point

    return experiment

def saveByLine(X, path, seqDim=4):
    with open(path, 'w') as f:
        for i in range(len(X)):
            line = ''
            for j in range(len(X[i])):
                if (j < 2*23*seqDim): line += "%d"   % X[i][j]+' ' # one-hot sequence
                else:                 line += "%.2f" % X[i][j]+' ' # epigenetics encodings (can be floats)
            f.write(line+'\n')
            
def normaliseCF(cleavage_freq, cutoff=1e-5, stdev=2.0, mean=0.0, doClip=True): # normalises the cleavage frequencies given as a 1D list to a Gaussian N(0, stdev^2) using a Box-Cox transformation, clipped at -2*stdev and 2*stdev
    # normalise in terms of maximum cleavage frequency
    cleavage_freq = [0 if x is None else x for x in cleavage_freq] # make sure all elements are floats (no Nones)
    maximum = np.nanmax(cleavage_freq)
    cleavage_freq = cleavage_freq / maximum # numpy arrays support these element-wise operations
    cleavage_freq = np.clip(cleavage_freq, cutoff, 1)

    # power law transformation of cleavage frequency
    pt = PowerTransformer(method='box-cox', standardize=True) # map to normal distribution with zero mean and unit variance
    cleavage_freq = cleavage_freq.reshape(-1, 1)  # fitting requires this shape for data points with only a single feature
    pt.fit(cleavage_freq)
    cleavage_freq = pt.transform(cleavage_freq)
    cleavage_freq = cleavage_freq + mean
    cleavage_freq = cleavage_freq * stdev  # adjust variance such that values >variancesq are more than one sigma above mean
    if doClip: cleavage_freq = np.clip(cleavage_freq, mean-2*stdev, mean+2*stdev) # cap distribution (without discarding values)
    
    return cleavage_freq.flatten(), pt                # undo reshape

def inverse_scale_CF(inp, variance=2.0, mean=0.0):
    variancesq = variance**2
    return inp*np.sqrt(1/variancesq)+mean