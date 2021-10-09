import numpy as np
import scipy.linalg as la

def divide_and_conquer(block_mats,tier_names = None,output=True):
    '''
    A script for computing the inverse matrix of a large matrix,
    divided into blocks in order to track progress.
    '''
    num_tiers = len(block_mats)
    
    inv_mats = [[] for j in range(num_tiers)]

    if tier_names == None:
        tier_names = np.arange(num_tiers).astype(str)

    for j in range(num_tiers):
        if output:
            print('Computing tier '+tier_names[j]+'...')
        if j == 0:
            inv = la.inv(np.eye(block_mats[j][j].shape[0]) - block_mats[j][j])
            inv_mats[0].append(inv)
        elif j > 0:
            mat = block_mats[j][j].copy()
            if output:
                print('\tComputing new inverse...')
            for k in range(j):
                for l in range(j):
                    mat += block_mats[j][k]@inv_mats[k][l]@block_mats[l][j]
            inv = la.inv(np.eye(block_mats[j][j].shape[0]) - mat)

            if output:
                print('\tAdding new columns and rows...')
            for k in range(j):
                # Add new column
                mat = np.zeros(block_mats[k][j].shape)
                for l in range(j):
                    mat += inv_mats[k][l]@block_mats[l][j]@inv
                inv_mats[k].append(mat)

                # Add new row
                mat = np.zeros(block_mats[j][k].shape)
                for l in range(j):
                    mat += inv@block_mats[j][l]@inv_mats[l][k]
                inv_mats[j].append(mat)

            new_mats = []
            if output:
                print('\tUpdating old entries...')
            for k in range(j):
            # Update old entries
                new_new_mats = []
                for l in range(j):
                    mat = inv_mats[k][l].copy()
                    for m in range(j):
                        mat += inv_mats[k][j]@block_mats[j][m]@inv_mats[m][l]
                    new_new_mats.append(mat)
                new_mats.append(new_new_mats)
            for k in range(j):
                for l in range(j):
                    inv_mats[k][l] = new_mats[k][l]

            # Add new diagonal entry
            inv_mats[j].append(inv)
    return inv_mats