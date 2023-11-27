import numpy as np
from tqdm import tqdm
from scipy.sparse import csc_array
import collections
import itertools

def GetData(filename):
    ''' Reads the user-movie data from a .npy file and returns it as a sparse matrix '''

    user_movie_data = np.load(filename)

    row = user_movie_data[:, 0] -1  # the users indices (user ID 1 has index 0)
    col = user_movie_data[:, 1] -1 # the movies indices 
    data = user_movie_data[:, 2] # the ratings
    data[data > 0] = 1 # binarize the data

    csc_user_movie_data = csc_array((data, (row, col)), shape=(np.max(row)+1, np.max(col)+1), dtype=np.int8)

    return csc_user_movie_data

def MinHash(csc_user_movie_data, num_permutations):
    ''' Performs minhashing on the user-movie data '''
    
    minhash_matrix = np.zeros((num_permutations, csc_user_movie_data.shape[0]))

    # for each permutation, find the first nonzero element for each user
    for i in tqdm(range(0, num_permutations)): 
        permutation_indices = np.random.permutation(csc_user_movie_data.shape[1])
        permuted_csc_user_movie_data = csc_user_movie_data[:,permutation_indices] # permuted columns

        # The following line is the bottleneck. Reduce the search space by only looking at the first x movies; the first nonzero value for each user is likely in this set
        try:
            users_nonzeroes, movies_nonzeroes = permuted_csc_user_movie_data[:,:1000].nonzero() # this gives the row and column indices of the nonzero elements
        except ValueError: # if the first nonzero value is not in the first 1000 movies, look at all movies
            users_nonzeroes, movies_nonzeroes = permuted_csc_user_movie_data[:,:].nonzero() 
        nonzero_user_indices = np.unique(users_nonzeroes, return_index=True)[1] 
        minhash_matrix[i,:] = movies_nonzeroes[nonzero_user_indices] # add first nonzero value for each user to minhash matrix

    return minhash_matrix

def FindCandidatePairs(minhash_matrix, b):
    ''' Finds candidate pairs using the LSH algorithm '''
    
    permutations, users = minhash_matrix.shape
    buckets = collections.defaultdict(set) # dictionary with empty sets as default values; sets are used to avoid duplicates
    bands = np.array_split(minhash_matrix, b, axis=0) # split the matrix into b bands
    
    for i,band in enumerate(bands):
        for j in range(users):
            # add the band number to the tuple so that the buckets are unique to the bands
            band_id = tuple(list(band[:,j])+[str(i)])
            buckets[band_id].add(j)

    candidate_pairs = set() # do not allow duplicate pairs
    for bucket in buckets.values():
        if len(bucket) > 1: # buckets with 1 user have no pairs
            for pair in itertools.combinations(bucket, 2):
                if (minhash_matrix[:,pair[0]] == minhash_matrix[:,pair[1]]).sum() / permutations > 0.49: # filter out pairs with low approximate similarity
                    candidate_pairs.add(pair)

    return candidate_pairs

def JaccardSimilarity(arr1, arr2):
    ''' Finds the Jaccard similarity between two arrays '''	
    intersection = np.intersect1d(arr1, arr2)
    union = np.union1d(arr1, arr2)
    return len(intersection) / len(union)

def FindSimilarities(csc_user_movie_data, candidate_pairs, txtfile='js.txt'):
    ''' Finds the Jaccard similarity between all candidate pairs '''

    with open(txtfile,'w') as f:
        f.write('') # empties file 

    csc_user_movie_data = csc_user_movie_data.tocsr() # convert to csr format for faster slicing

    for pair in tqdm(candidate_pairs):
        similarity_real = JaccardSimilarity( csc_user_movie_data[[pair[0]],:].nonzero()[1], csc_user_movie_data[[pair[1]],:].nonzero()[1] )

        if similarity_real > 0.5:
            sorted_pair = tuple(sorted(pair))
            # write to text file
            with open(txtfile, 'a') as f:
                f.write(str(sorted_pair[0]+1) + ',' + str(sorted_pair[1]+1) + '\n')

def main(args):
    ''' Main function
    args: (namespace), command line arguments
    
    output: similar_pairs.txt, a list of similar pairs of users
    returns:  nothing'''

    np.random.seed(args.s) #set the random seed

    #import the data
    csc_user_movie_data = GetData(args.d)

    #run parameters
    num_permutations = 150 

    #run LSH
    minhash_matrix = MinHash(csc_user_movie_data, num_permutations)
    candidate_pairs = FindCandidatePairs(minhash_matrix, b=num_permutations//5)

    #calculate the real similarity
    FindSimilarities(csc_user_movie_data, candidate_pairs, 'js.txt')