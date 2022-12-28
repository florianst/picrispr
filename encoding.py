from load_data import normaliseCF, seqToOneHot, turnToOneHot, seqToDinucleotides
from picrispr import FeatureEncoding, splitBpwise

def noSequence(experiment, augmented, cutoff, CFtoScore=normaliseCF, seqToOneHot=seqToOneHot, count=None, i=0, featurenames=None, numBpWise=0, *args, **kwargs):
    encoding = FeatureEncoding(epiDim=len(experiment[0])-1, epiStart=0, seqDim=0, featurenames=featurenames, encodingFunction="noSequence")
    if (count == 1): print("no sequence encoding")

    cleavage_freq = [experiment[datapoint][-1] for datapoint in range(len(experiment))]  # extract cleavage_freq
    cleavage_freq_transformed, pt = CFtoScore(cleavage_freq, cutoff)
    for line in range(len(experiment)):
        experiment[line] = [item for item in experiment[line]]  # fetchall gives a list of tuples, need to convert into list of lists
        experiment[line][-1] = cleavage_freq_transformed[line]  # write cleavage frequency back to experiment array

    for key in range(len(experiment)):
        bpwiseFeatures = experiment[key][:numBpWise]
        bpwiseList = []
        for bpwise in bpwiseFeatures:
            if bpwise is not None:
                bpwise = splitBpwise(bpwise)
                if len(bpwise) == 23: bpwiseList.append(bpwise)
                else: bpwiseList.append([0] * 23)  # in case nucleosome data is not available
            else: bpwiseList.append([0] * 23)
        bpwiseList = [item for sublist in bpwiseList for item in sublist]
        experiment[key] = bpwiseList + [float(i) if i is not None else 0 for i in experiment[key][numBpWise:]]

    for key in range(len(augmented)):
        bpwiseFeatures = augmented[key][:numBpWise]
        bpwiseList = []
        for bpwise in bpwiseFeatures:
            if bpwise is not None:
                bpwise = splitBpwise(bpwise)
                if len(bpwise) == 23: bpwiseList.append(bpwise)
                else: bpwiseList.append([0] * 23)  # in case nucleosome data is not available
            else: bpwiseList.append([0] * 23)
        bpwiseList = [item for sublist in bpwiseList for item in sublist]
        augmented[key] = bpwiseList + [float(i) if i is not None else 0 for i in augmented[key][numBpWise:]]

    return experiment, augmented, encoding, pt

def noEncoding(experiment, augmented, cutoff, CFtoScore=normaliseCF, seqToOneHot=seqToOneHot, count=None, i=0, featurenames=None, numBpWise=0, *args, **kwargs):
    encoding = FeatureEncoding(epiDim=len(experiment[0])-2, epiStart=0, seqDim=0, featurenames=featurenames, encodingFunction="noEncoding")

    cleavage_freq = [experiment[datapoint][-1] for datapoint in range(len(experiment))]  # extract cleavage_freq
    cleavage_freq_transformed, pt = CFtoScore(cleavage_freq, cutoff)
    for line in range(len(experiment)):
        experiment[line] = [item for item in experiment[line]]  # fetchall gives a list of tuples, need to convert into list of lists
        experiment[line][-1] = cleavage_freq_transformed[line]  # write cleavage frequency back to experiment array

    for key in range(len(experiment)):
        bpwiseFeatures = experiment[key][:numBpWise]
        bpwiseList = []
        for bpwise in bpwiseFeatures:
            if bpwise is not None:
                bpwise = splitBpwise(bpwise)
                if len(bpwise) == 23: bpwiseList.append(bpwise)
                else: bpwiseList.append([0] * 23)  # in case nucleosome data is not available
            else: bpwiseList.append([0] * 23)
        bpwiseList = [item for sublist in bpwiseList for item in sublist]
        experiment[key] = bpwiseList + [i if i is not None else 0 for i in experiment[key][numBpWise:]]

    for key in range(len(augmented)):
        bpwiseFeatures = augmented[key][:numBpWise]
        bpwiseList = []
        for bpwise in bpwiseFeatures:
            if bpwise is not None:
                bpwise = splitBpwise(bpwise)
                if len(bpwise) == 23: bpwiseList.append(bpwise)
                else: bpwiseList.append([0] * 23)  # in case nucleosome data is not available
            else: bpwiseList.append([0] * 23)
        bpwiseList = [item for sublist in bpwiseList for item in sublist]
        augmented[key] = bpwiseList + [i if i is not None else 0 for i in augmented[key][numBpWise:]]

    return experiment, augmented, encoding, pt

def oneHotSingleNuclTargetMismatchType(experiment, augmented, cutoff, CFtoScore=normaliseCF, seqToOneHot=seqToOneHot, count=None, i=0, featurenames=None, numBpWise=0, *args, **kwargs):
    if featurenames is None: featurenames=['A_match', 'T_match', 'C_match', 'G_match', 
                                           'A_mismT', 'T_mismC', 'C_mismG', 'G_mismA', 
                                           'A_mismC', 'T_mismG', 'C_mismA', 'G_mismT',
                                           'A_mismG', 'T_mismA', 'C_mismT', 'G_mismC',
                                           'CTCF', 'DNase', 'RRBS', 'H3K4me3', 'DRIP', 'energy1', 'energy2', 'energy3', 'energy4', 'energy5']
    epiStart, seqDim = 16, 16
    
    #                                                    2 for sequences, 1 for cleavage_freq
    encoding = FeatureEncoding(epiDim=len(experiment[0])-2-1, epiStart=epiStart, seqDim=seqDim, featurenames=featurenames, encodingFunction="oneHotSingleNuclTargetMismatchType")

    cleavage_freq = [experiment[datapoint][-1] for datapoint in range(len(experiment))] # extract cleavage_freq
    cleavage_freq_transformed, pt = CFtoScore(cleavage_freq, cutoff)
    for line in range(len(experiment)):
        experiment[line] = [item for item in experiment[line]] # fetchall gives a list of tuples, need to convert into list of lists
        experiment[line][-1] = cleavage_freq_transformed[line] # write cleavage frequency back to experiment array
    
    # turn both grna and target sequence into 23*4 long one-hot vectors
    experiment = turnToOneHot(experiment, seqToOneHot=seqToOneHot, encodingFunction="oneHotSingleNuclTargetMismatchType", numBpWise=numBpWise)
    augmented  = turnToOneHot(augmented,  seqToOneHot=seqToOneHot, encodingFunction="oneHotSingleNuclTargetMismatchType", numBpWise=numBpWise)
    
    return experiment, augmented, encoding, pt       
        
def oneHotSingleNuclTargetMismatch(experiment, augmented, cutoff, CFtoScore=normaliseCF, seqToOneHot=seqToOneHot, count=None, i=0, featurenames=None, numBpWise=0, *args, **kwargs):
    if featurenames is None: featurenames=['A', 'A_mism', 'T', 'T_mism', 'C', 'C_mism', 'G', 'G_mism', 'CTCF', 'DNase', 'RRBS', 'H3K4me3', 'DRIP', 'energy1', 'energy2', 'energy3', 'energy4', 'energy5']
    epiStart, seqDim = 8, 8
    
    #                                                    2 for sequences, 1 for cleavage_freq
    encoding = FeatureEncoding(epiDim=len(experiment[0])-2-1, epiStart=epiStart, seqDim=seqDim, featurenames=featurenames, encodingFunction="oneHotSingleNuclTargetMismatch")
    
    cleavage_freq = [experiment[datapoint][-1] for datapoint in range(len(experiment))] # extract cleavage_freq
    cleavage_freq_transformed, pt = CFtoScore(cleavage_freq, cutoff)
    for line in range(len(experiment)):
        experiment[line] = [item for item in experiment[line]] # fetchall gives a list of tuples, need to convert into list of lists
        experiment[line][-1] = cleavage_freq_transformed[line] # write cleavage frequency back to experiment array
    
    # turn both grna and target sequence into 23*4 long one-hot vectors
    experiment = turnToOneHot(experiment, seqToOneHot=seqToOneHot, encodingFunction="oneHotSingleNuclTargetMismatch", numBpWise=numBpWise)
    augmented  = turnToOneHot(augmented,  seqToOneHot=seqToOneHot, encodingFunction="oneHotSingleNuclTargetMismatch", numBpWise=numBpWise)
    
    return experiment, augmented, encoding, pt

def oneHotSingleNucl(experiment, augmented, cutoff, CFtoScore=normaliseCF, seqToOneHot=seqToOneHot, count=None, i=0, featurenames=None, numBpWise=0, CRISPRNetStyle=False, *args, **kwargs):
    if featurenames is None: featurenames=['A', 'T', 'C', 'G', 'CTCF', 'DNase', 'RRBS', 'H3K4me3', 'DRIP', 'energy1', 'energy2', 'energy3', 'energy4', 'energy5']
    epiStart, seqDim = 4, 4

    if CRISPRNetStyle:
        epiStart, seqDim = 6, 6
        featurenames = featurenames[:4] + ['base order desc', 'base order asc'] + featurenames[4:]
    
    #                                                    2 for sequences, 1 for cleavage_freq
    encoding = FeatureEncoding(epiDim=len(experiment[0])-2-1, epiStart=epiStart, seqDim=seqDim, featurenames=featurenames, encodingFunction="oneHotSingleNucl")
    
    cleavage_freq = [experiment[datapoint][-1] for datapoint in range(len(experiment))] # extract cleavage_freq
    cleavage_freq_transformed, pt = CFtoScore(cleavage_freq, cutoff)
    for line in range(len(experiment)):
        experiment[line] = [item for item in experiment[line]] # fetchall gives a list of tuples, need to convert into list of lists
        experiment[line][-1] = cleavage_freq_transformed[line] # write cleavage frequency back to experiment array
    
    # turn both grna and target sequence into 23*4 long one-hot vectors
    experiment = turnToOneHot(experiment, seqToOneHot=seqToOneHot, encodingFunction="oneHotSingleNucl", numBpWise=numBpWise, CRISPRNetStyle=CRISPRNetStyle)
    augmented  = turnToOneHot(augmented,  seqToOneHot=seqToOneHot, encodingFunction="oneHotSingleNucl", numBpWise=numBpWise, CRISPRNetStyle=CRISPRNetStyle)
    
    return experiment, augmented, encoding, pt

def oneHotDinuclTargetMismatch(experiment, augmented, cutoff, CFtoScore=normaliseCF, seqToOneHot=seqToDinucleotides, count=None, i=0, featurenames=None, numBpWise=0, *args, **kwargs):
    featurenames = ['AA', 'AA_mism', 'AT', 'AT_mism', 'AC', 'AC_mism', 'AG', 'AG_mism', 
                    'TA', 'TA_mism', 'TT', 'TT_mism', 'TC', 'TC_mism', 'TG', 'TG_mism', 
                    'CA', 'CA_mism', 'CT', 'CT_mism', 'CC', 'CC_mism', 'CG', 'CG_mism', 
                    'GA', 'GA_mism', 'GT', 'GT_mism', 'GC', 'GC_mism', 'GG', 'GG_mism', 'CTCF', 'DNase', 'RRBS', 'DRIP', 'energy_2', 'energy_3', 'energy_5']
    epiStart, seqDim = 32, 32
    #                                                    2 for sequences, 1 for cleavage_freq
    encoding = FeatureEncoding(epiDim=len(experiment[0])-2-1, epiStart=epiStart, seqDim=seqDim, featurenames=featurenames, encodingFunction="oneHotDinuclTargetMismatch")
    if (count == 1): print("one-hot dinucleotide encoding in interface mode")
    
    cleavage_freq = [experiment[datapoint][-1] for datapoint in range(len(experiment))] # extract cleavage_freq
    cleavage_freq_transformed, pt = CFtoScore(cleavage_freq, cutoff)
    for line in range(len(experiment)):
        experiment[line] = [item for item in experiment[line]] # fetchall gives a list of tuples, need to convert into list of lists
        experiment[line][-1] = cleavage_freq_transformed[line] # write cleavage frequency back to experiment array
    
    # turn both grna and target sequence into 23*4 long one-hot vectors
    experiment = turnToOneHot(experiment, seqToOneHot=seqToOneHot, numBpWise=numBpWise)


    augmented = turnToOneHot(augmented, seqToOneHot=seqToOneHot, numBpWise=numBpWise)

    
    return experiment, augmented, encoding, pt

def oneHotDinucl(experiment, augmented, cutoff, CFtoScore=normaliseCF, seqToOneHot=seqToDinucleotides, count=None, i=0, featurenames=None, numBpWise=0, *args, **kwargs):
    featurenames = ['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG', 'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG', 'CTCF', 'DNase', 'RRBS', 'DRIP', 'energy_2', 'energy_3', 'energy_5']
    epiStart, seqDim = 16, 16
    #                                                    2 for sequences, 1 for cleavage_freq
    encoding = FeatureEncoding(epiDim=len(experiment[0])-2-1, epiStart=epiStart, seqDim=seqDim, featurenames=featurenames, encodingFunction="oneHotDinucl")
    if (count == 1): print("one-hot dinucleotide encoding")
    
    cleavage_freq = [experiment[datapoint][-1] for datapoint in range(len(experiment))] # extract cleavage_freq
    cleavage_freq_transformed, pt = CFtoScore(cleavage_freq, cutoff)
    for line in range(len(experiment)):
        experiment[line] = [item for item in experiment[line]] # fetchall gives a list of tuples, need to convert into list of lists
        experiment[line][-1] = cleavage_freq_transformed[line] # write cleavage frequency back to experiment array
    
    # turn both grna and target sequence into 23*4 long one-hot vectors
    experiment = turnToOneHot(experiment, seqToOneHot=seqToOneHot, numBpWise=numBpWise)

    augmented = turnToOneHot(augmented, seqToOneHot=seqToOneHot, numBpWise=numBpWise)
    
    return experiment, augmented, encoding, pt