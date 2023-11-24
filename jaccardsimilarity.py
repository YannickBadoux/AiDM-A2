import numpy as np
from tqdm import tqdm
from scipy.sparse import csc_array
import collections
import itertools
import matplotlib.pyplot as plt

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

    r = permutations//b
    print(f'The cutoff is {(1/b)**(1/r):.3f}')
    
    for i,band in enumerate(bands):
        for j in range(users):
            # add the band number to the tuple so that the buckets are unique to the bands
            band_id = tuple(list(band[:,j])+[str(i)])
            buckets[band_id].add(j)

    candidate_pairs = set() # do not allow duplicate pairs
    for bucket in buckets.values():
        if len(bucket) > 1: # buckets with 1 user have no pairs
            for pair in itertools.combinations(bucket, 2):
                candidate_pairs.add(pair)

    return candidate_pairs

def JaccardSimilarity(arr1, arr2):
    ''' Finds the Jaccard similarity between two arrays '''	
    intersection = np.intersect1d(arr1, arr2)
    union = np.union1d(arr1, arr2)
    return len(intersection) / len(union)

def FindSimilarities(csc_user_movie_data, candidate_pairs, txtfile='similar_pairs.txt'):
    ''' Finds the Jaccard similarity between all candidate pairs '''

    similar_pairs = set()
    real_sims = []

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
            similar_pairs.add(sorted_pair)
            real_sims.append(similarity_real)

    print("Number of similar pairs: {}".format(len(real_sims)))

    return real_sims, candidate_pairs



if __name__ == '__main__':
    np.random.seed(21)

    filename = 'C:/Users/stijn/Documents/jaar 2/AIDM/assignment 2/user_movie_rating.npy'
    csc_user_movie_data = GetData(filename)


    for num_permutations in [105,154,182]:
        simsplot = []
        for i in range(3): # run 3 times to get average results
            simsplot.append([])
            # optimise number of permutations and bands with a loop since this is fast anyway.
            num_permutations = num_permutations
            minhash_matrix = MinHash(csc_user_movie_data, num_permutations)

            candidate_pairs = FindCandidatePairs(minhash_matrix, b=num_permutations//7)  # b=20
            print("Number of candidate pairs: {} with {} users".format(len(candidate_pairs), minhash_matrix.shape[1]))

            real_sims, similar_pairs = FindSimilarities(csc_user_movie_data, candidate_pairs)
            simsplot[i].append(real_sims)
        
        lengths = []
        simsplot_sorted = []
        for i, simlist in enumerate(simsplot):
            simsplot[i] = np.array(simlist).flatten()
            lengths.append(len(simsplot))
        arg = np.argsort(lengths)
        
        for i in arg:
            simsplot_sorted.append(simsplot[i])

        plt.figure(figsize=(8,6))
        plt.boxplot(simsplot_sorted, vert=False, positions=(0.5,1,1.5))
        plt.yticks([0.5,1,1.5],["N$_{pairs}$ = "+str(len(simsplot_sorted[i])) for i in range(len(simsplot_sorted))], fontsize=16)
        plt.ylim([0.3,1.7])
        plt.xticks(fontsize=14)
        plt.xlabel('Jaccard similarity', fontsize=16)
        plt.title(f'Jaccard Similarity Distribution using {num_permutations} signatures', fontsize=18)
        plt.savefig(f'jaccard_{num_permutations}.pdf', bbox_inches='tight')	