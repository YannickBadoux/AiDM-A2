import numpy as np
from scipy.sparse import csc_array
import itertools
from tqdm import tqdm
import collections
import time

def ImportData(path, measure):
    ''' Reads the data from a .npy file which can be found at path and returns a sparse matrix. '''
    data = np.load(path)
    row = data[:, 0] - 1 #user indices
    col = data[:, 1] - 1 #movie indces
    if measure == 'dcs':
        rating = np.clip(data[:, 2],1,1) #set all the ratings to 1 
    else:
        rating = data[:, 2] #ratings

    sparse = csc_array((rating, (col, row)), shape=(np.max(col)+1, np.max(row)+1)) #make a sparse matrix
    return sparse

def RandomProjections(ratings, len_sig):
    ''' Make signatures using random projections for the sparse matrix of ratings. '''
    hp = np.random.uniform(-1, 1, size=(len_sig, np.shape(ratings)[0])) #len_sig normal vectors of length movies
    A = hp@ratings 
    return np.where(A>0, 1, 0) #Signature matrix

def BandingTechnique(sign, num_row, len_sig):
    ''' Using the banding technique on signatures 'sign' where 'len_sig'
    is the length of the signatures and 'num_row' the number of rows used 
    per band. It returns a numpy array which contains len_sig/num_row
    separate numpy arrays which are the bands. '''
    num_bands = len_sig/num_row #calculate the number of bands
    return np.array_split(sign, num_bands, axis=0) #split the signatures into bands

def BinarySignatures(band_sig,  num_row):
    ''' This function converts binary bands to decimal numbers. '''
    binary_column = 2**np.arange(num_row).reshape(1, -1)
    bin_band_sig = []

    for i in range(len(band_sig)):
        bin_band_sig.append(np.hstack(binary_column@band_sig[i]))
    return np.array(bin_band_sig)

def MakeBuckets(sign):
    ''' Puts similar signatures into buckets. '''
    bucket_dict = collections.defaultdict(set) #This makes sure that we have no double buckets

    for band in range(np.shape(sign)[0]):
        for user, value in enumerate(sign[band,:]):
            bucket_dict[f"b{band}v{value}"].add(user)
    return bucket_dict

def CandidatePairs(buckets, sign):
    ''' Make a list of candidate pairs from the different buckets. '''
    pairs = set()
    for key, bucket in buckets.items():
        if len(bucket) > 1:
            for pair in itertools.combinations(bucket, 2):
                    if (sign[:,pair[0]] == sign[:,pair[1]]).sum()/np.shape(sign)[0] >= 0.72:
                        pairs.add(tuple(sorted(pair)))
    return pairs

def CosineSimilarity(u1, u2):
    ''' Calculates the cosine similarity between user 1 and user 2 which
    are both column vectors in a sparse matrix. Returns a float.'''
    costheta = (np.transpose(u1)@u2)/(np.sqrt(np.transpose(u1)@u1) * np.sqrt(np.transpose(u2)@u2))
    return float((1-(np.arccos(costheta)/np.pi))[0][0])

def FindSimilarPairs(data, pairs, end, measure):
    ''' Finds the similar pairs using the cosine similarity with a threshold of 0.73 and 
    writes the pairs to a file. '''

    sim_pairs = []
    sims = []

    with open(f'{measure}.txt', 'w') as f:
        f.write('') #Empties file

    for (u1, u2) in tqdm(pairs):
        if (time.time() >= end):
            break #Make sure it does not run longer than 30 minutes
        sim = CosineSimilarity(data[:,[u1]], data[:,[u2]])
        if sim > 0.73:
            with open(f'{measure}.txt', 'a') as f:
                f.write(str(sorted((u1+1, u2+1)))+'\n')
            sim_pairs.append((u1, u2))
            sims.append(sim)
    print(f'Found {len(sim_pairs)} similar pairs')
    return (sims, sim_pairs)


def main(args):
        '''Main function which runs the whole program.
        args: (namespace) arguments from the command line
        
        output: similar_pairs.txt, a list of similar pairs of users
        returns: nothing'''

        #set the random seed
        np.random.seed(args.s)
        #Importing the data
        data = ImportData(args.d, args.m)

        #Parameters
        nrow = 18
        if args.m == 'dcs':
            nsig = 180
        elif args.m == 'cs':
            nsig = 144

        #time
        start = time.time()
        runtime = 30*60 # maximum runtime in seconds
        end  = start + runtime

        #Creating signatures 
        sig = RandomProjections(data, len_sig= nsig)
        sig_b = BandingTechnique(sig, num_row = nrow, len_sig = nsig)
        sig_bin = BinarySignatures(sig_b, num_row = nrow)

        #Creating the buckets
        bucket_dict = MakeBuckets(sig_bin)
        cand_pairs = CandidatePairs(bucket_dict, sig)

        #Finding similar pairs
        FindSimilarPairs(data, cand_pairs, end, args.m)