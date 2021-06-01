import numpy as np
from stoclust.Group import Group
from stoclust.Aggregation import Aggregation
from stoclust.utils import block_sum
from stoclust.utils import stoch
from scripts import GTAP
from scipy.special import kl_div
import os

'''
Useful scripts for constructing large
block matrices representing GTAP data,
or generating null model block matrices
based on the GTAP data. These block matrices
are used for the inversion step in Leontief
analysis.

Requires:
    stoclust
    numpy
    scipy
    tqdm
'''

'''
`names` gives the labels of the four major sectors:
    D: domestic commodities
    M: imported commodities
    F: factors of production
    C: final demand/consumption
'''

names = ['D','M','F','C','H']

def null_blocks(num_samples = 1000,div_factor = 1,name='null'):
    '''
        Generates a 4x4 list of lists containing the coefficients of the null model.
        The list elements are matrices connecting the different types of sector
        (a ----- denotes a zero block where there are no connections):

        [
            [ D x D, D x M, -----, D x C ],
            [ M x D, -----, -----, M x C ],
            [ F x D, -----, -----, ----- ],
            [ -----, -----, -----, ----- ],
        ]

        Saves each block to a numpy array file. For instance, the D x D block goes to 
        `scripts/data/computed/blocks/null/DD_T.npy`.
    '''

    # Initialize important parameters
    gtap = GTAP()
    num_reg = gtap.regions.items.size
    num_commod = gtap.commodities.items.size
    num_fac = gtap.factors.items.size

    new_reg = gtap.megaregions.at_scale(1)
    new_commod = gtap.commodities.at_scale(2)
    new_fac = gtap.factors.at_scale(2)

    num_new_reg = new_reg.clusters.size
    num_new_commod = new_commod.clusters.size
    num_new_fac = new_fac.clusters.size

    # Pull world-averaged technical coefficients
    MD_full = gtap.fetch_sam(demand=gtap.sectors['activities'],supply=gtap.sectors['imports'])
    DD_full = gtap.fetch_sam(demand=gtap.sectors['activities'],supply=gtap.sectors['domestic'])

    DC_full = gtap.fetch_sam(demand=gtap.sectors['final_demand'],supply=gtap.sectors['domestic'])
    MC_full = gtap.fetch_sam(demand=gtap.sectors['final_demand'],supply=gtap.sectors['imports'])

    MD_agg = gtap.commodities.measure(gtap.commodities.measure(MD_full,axis=1)[:,new_commod.clusters.in_superset],axis=2)[:,:,new_commod.clusters.in_superset]
    DD_agg = gtap.commodities.measure(gtap.commodities.measure(DD_full,axis=1)[:,new_commod.clusters.in_superset],axis=2)[:,:,new_commod.clusters.in_superset]
    DC_agg = gtap.commodities.measure(DC_full,axis=1)[:,new_commod.clusters.in_superset]
    MC_agg = gtap.commodities.measure(MC_full,axis=1)[:,new_commod.clusters.in_superset]

    # Compute average divergence from mean
    P = stoch(MD_agg+DD_agg,axis=(0,1,2))
    P0 = P.sum(axis=0)
    Q = stoch(P.sum(axis=1),axis=0)
    C_world = stoch(P0,axis=0)
    C_div = kl_div(P,P0[None,:,:]*Q[:,None,:]).sum(axis=(0,1))/P.sum(axis=(0,1))

    P = stoch(MC_agg+DC_agg,axis=(0,1,2))
    P0 = P.sum(axis=0)
    Q = stoch(P.sum(axis=1),axis=0)
    D_world = stoch(P0,axis=0)
    D_div = kl_div(P,P0[None,:,:]*Q[:,None,:]).sum(axis=(0,1))/P.sum(axis=(0,1))

    # Generate null technical coefficients
    alphas = C_world/C_div[None,:]
    size = num_samples*num_new_reg
    C_nulls = []
    for c in range(new_commod.clusters.size):
        C_nulls.append(np.random.dirichlet(alphas[:,c]*div_factor,size=size).reshape([num_samples,num_new_reg,num_new_commod]))
    tech_null = np.moveaxis(np.array(C_nulls),[0,1,2,3],[3,0,1,2])

    alphas = D_world/D_div[None,:]
    size = num_samples*num_new_reg
    con_nulls = []
    for c in range(3):
        con_nulls.append(np.random.dirichlet(alphas[:,c]*div_factor,size=size).reshape([num_samples,num_new_reg,num_new_commod]))
    con_null = np.moveaxis(np.array(con_nulls),[0,1,2,3],[3,0,1,2])

    # Generate "social" coefficients randomly
    C_imp = np.moveaxis(
        np.random.dirichlet(
            (1,1),size=num_samples*num_new_reg*3
        ).reshape([num_samples,num_new_reg,3,2]),
        [0,1,2,3],[0,1,3,2]
    )
    V_add = np.moveaxis(
        np.random.dirichlet(
            (1,1),size=num_samples*num_new_reg*num_new_commod
        ).reshape([num_samples,num_new_reg,num_new_commod,2]),
        [0,1,2,3],[0,1,3,2]
    )
    D_imp = np.moveaxis(
        np.random.dirichlet(
            (1,1),size=num_samples*num_new_reg*num_new_commod
        ).reshape([num_samples,num_new_reg,num_new_commod,2]),
        [0,1,2,3],[0,1,3,2]
    )
    M_reg = np.moveaxis(
        np.random.dirichlet(
            (1,)*num_new_reg,size=num_samples*num_new_reg*num_new_commod
        ).reshape([num_samples,num_new_reg,num_new_commod,num_new_reg]),
        [0,1,2,3],[0,3,2,1]
    )
    D_fac = np.moveaxis(
        np.random.dirichlet(
            (1,)*num_new_fac,size=num_samples*num_new_reg*num_new_commod
        ).reshape([num_samples,num_new_reg,num_new_commod,num_new_fac]),
        [0,1,2,3],[0,1,3,2]
    )

    # Construct blocks
    DD_T = ((V_add[:,:,0,None,:]*D_imp[:,:,0,None,:]*tech_null[:,:,:,:])[:,:,:,None,:]*np.eye(num_new_reg)[None,:,None,:,None]).reshape([num_samples,num_new_reg*num_new_commod,num_new_reg*num_new_commod])
    MD_T = ((V_add[:,:,0,None,:]*D_imp[:,:,1,None,:]*tech_null[:,:,:,:])[:,:,:,None,:]*np.eye(num_new_reg)[None,:,None,:,None]).reshape([num_samples,num_new_reg*num_new_commod,num_new_reg*num_new_commod])
    FD_T = ((V_add[:,:,1,None,:]*D_fac[:,:,:,:])[:,:,:,None,:]*np.eye(num_new_reg)[None,:,None,:,None]).reshape([num_samples,num_new_reg*num_new_fac,num_new_reg*num_new_commod])

    DC_T = ((C_imp[:,:,0,None,:]*con_null[:,:,:,:])[:,:,:,None,:]*np.eye(num_new_reg)[None,:,None,:,None]).reshape([num_samples,num_new_reg*num_new_commod,num_new_reg*3])
    MC_T = ((C_imp[:,:,1,None,:]*con_null[:,:,:,:])[:,:,:,None,:]*np.eye(num_new_reg)[None,:,None,:,None]).reshape([num_samples,num_new_reg*num_new_commod,num_new_reg*3])

    DX_T = (M_reg[:,:,:,:,None]*np.eye(num_new_commod)[None,None,:,None,:]).reshape([num_samples,num_new_reg*num_new_commod,num_new_reg*num_new_commod])

    if not os.path.isdir(os.path.abspath('scripts/data/computed/blocks')):
        os.mkdir(os.path.abspath('scripts/data/computed/blocks'))
    if not os.path.isdir(os.path.abspath('scripts/data/computed/blocks/'+name)):
        os.mkdir(os.path.abspath('scripts/data/computed/blocks/'+name))
    
    blocks = [
        [DD_T,                   DX_T,                   np.zeros(FD_T.shape[0::2]+(FD_T.shape[1],)), DC_T],
        [MD_T,                   np.zeros(DD_T.shape),   np.zeros(FD_T.shape[0::2]+(FD_T.shape[1],)), MC_T],
        [FD_T,                   np.zeros(FD_T.shape),   np.zeros((FD_T.shape[0],)+(FD_T.shape[1],)*2),     np.zeros([num_samples,num_new_reg*num_new_fac,num_new_reg*3])],
        [np.zeros(DC_T.shape[0::2]+(DC_T.shape[1],)), np.zeros(DC_T.shape[0::2]+(DC_T.shape[1],)), np.zeros([num_samples,num_new_reg*3,num_new_reg*num_new_fac]), np.zeros([num_samples,num_new_reg*3,num_new_reg*3])],
    ]

    for i in range(4):
        for k in range(4):
            np.save('scripts/data/computed/blocks/'+name+'/'+names[i]+names[k]+'_T.npy',blocks[i][k])

def industry_blocks(commodity_agg=None,region_agg=None,factor_agg=None,demand_agg=None):
    '''
        Pulls a 4x4 list of lists containing the coefficients of the GTAP 8 model.
        This method can receive stoclust Aggregations to specify whether regions,
        factors, commodities and demands should be aggregated before calculating
        the blocks. Aggregation results in smaller matrices and faster calculations,
        but less precision.

        The list elements are matrices connecting the different types of sector
        (a ----- denotes a zero block where there are no connections):

        [
            [ D x D, D x M, -----, D x C ],
            [ M x D, -----, -----, M x C ],
            [ F x D, -----, -----, ----- ],
            [ -----, -----, -----, ----- ],
        ]

        Saves each block to a numpy array file. For instance, the D x D block goes to 
        `scripts/data/computed/blocks/null/DD_T.npy`.
    '''
    
    gtap = GTAP()
    
    if commodity_agg is None:
        commodity_agg = gtap.commodities.at_scale(0)
    if region_agg is None:
        region_agg = gtap.regions.at_scale(0)
    if factor_agg is None:
        factor_agg = gtap.factors.at_scale(0)
    if demand_agg is None:
        demand_agg = Aggregation(
            Group(['PRIV','GOVT','CGDS']),
            Group(['PRIV','GOVT','CGDS']),
            {0: np.array([0]),
             1: np.array([1]),
             2: np.array([2])}
        )
    
    c_list = [list(g.in_superset) for c,g in commodity_agg]
    r_list = [list(g.in_superset) for r,g in region_agg]
    f_list = [list(g.in_superset) for f,g in factor_agg]
    d_list = [list(g.in_superset) for d,g in demand_agg]

    num_reg = len(r_list)
    num_com = len(c_list)
    num_fac = len(f_list)
    num_dem = len(d_list)
    
    print('Domestic inputs...')
    DIND = gtap.fetch_sam(supply=gtap.sectors['domestic'],demand=gtap.sectors['activities'])
    DD_agg = block_sum(
        block_sum(
            block_sum(
                DIND,
                c_list,axis=2
            ),c_list,axis=1
        ),r_list,axis=0
    )
    DD_mat = (DD_agg[:,:,None,:]*np.eye(num_reg)[:,None,:,None]).reshape([num_reg*num_com,num_reg*num_com])
    DD_mat = DD_mat - np.diag(np.diag(DD_mat))

    MIND = gtap.fetch_sam(supply=gtap.sectors['imports'],demand=gtap.sectors['activities'])
    MD_agg = block_sum(
        block_sum(
            block_sum(
                MIND,
                c_list,axis=2
            ),c_list,axis=1
        ),r_list,axis=0
    )
    MD_mat = (MD_agg[:,:,None,:]*np.eye(num_reg)[:,None,:,None]).reshape([num_reg*num_com,num_reg*num_com])

    FIND = gtap.fetch_sam(supply=gtap.sectors['factors'],demand=gtap.sectors['activities'])
    FD_agg = block_sum(
        block_sum(
            block_sum(
                FIND,
                c_list,axis=2
            ),f_list,axis=1
        ),r_list,axis=0
    )
    FD_mat = (FD_agg[:,:,None,:]*np.eye(num_reg)[:,None,:,None]).reshape([num_reg*num_fac,num_reg*num_com])

    print('Import inputs...')
    XM = gtap.fetch_sam(supply=gtap.sectors['domestic'],demand=gtap.sectors['trade'])
    XM_agg = block_sum(
        block_sum(
            block_sum(
                XM,
                r_list,axis=2
            ),c_list,axis=1
        ),r_list,axis=0
    )
    XM_mat = (XM_agg[:,:,:,None]*np.eye(num_com)[None,:,None,:]).reshape([num_reg*num_com,num_reg*num_com])

    XG = gtap.fetch_sam(supply=gtap.sectors['domestic'],demand=gtap.sectors['margin_incomes']).reshape([gtap.regions.items.size*gtap.commodities.items.size,3])
    GM = np.stack([
        gtap.fetch_sam(supply=gtap.sectors['land_margins'],demand=gtap.sectors['imports'])[:,0,:],
        gtap.fetch_sam(supply=gtap.sectors['water_margins'],demand=gtap.sectors['imports'])[:,0,:],
        gtap.fetch_sam(supply=gtap.sectors['air_margins'],demand=gtap.sectors['imports'])[:,0,:]
    ]).reshape([3,gtap.regions.items.size*gtap.commodities.items.size])
    XGM = (XG@stoch(GM)).reshape([gtap.regions.items.size,
                                  gtap.commodities.items.size,
                                  gtap.regions.items.size,
                                  gtap.commodities.items.size])
    XGM_agg = block_sum(
        block_sum(
            block_sum(
                block_sum(
                    XGM,c_list,axis=3
                ),r_list,axis=2
            ),c_list,axis=1
        ),r_list,axis=0
    )
    XM_mat = XM_mat + XGM_agg.reshape([num_reg*num_com,num_reg*num_com])

    print('Demand inputs...')
    DDEM = gtap.fetch_sam(supply=gtap.sectors['domestic'],demand=gtap.sectors['final_demand'])
    DDEM_agg = block_sum(
        block_sum(
            block_sum(
                DDEM,
                d_list,axis=2
            ),c_list,axis=1
        ),r_list,axis=0
    )
    DDEM_mat = (DDEM_agg[:,:,None,:]*np.eye(num_reg)[:,None,:,None]).reshape([num_reg*num_com,num_reg*num_dem])

    MDEM = gtap.fetch_sam(supply=gtap.sectors['imports'],demand=gtap.sectors['final_demand'])
    MDEM_agg = block_sum(
        block_sum(
            block_sum(
                MDEM,
                d_list,axis=2
            ),c_list,axis=1
        ),r_list,axis=0
    )
    MDEM_mat = (MDEM_agg[:,:,None,:]*np.eye(num_reg)[:,None,:,None]).reshape([num_reg*num_com,num_reg*num_dem])

    print('Stochastizing...')
    DOUT = np.sum(DD_mat,axis=0)+np.sum(MD_mat,axis=0)+np.sum(FD_mat,axis=0)
    MOUT = np.sum(XM_mat,axis=0)
    DEMOUT = np.sum(DDEM_mat,axis=0)+np.sum(MDEM_mat,axis=0)

    DD_T = DD_mat/(DOUT+(DOUT==0))[None,:]
    DDEM_T = DDEM_mat/(DEMOUT+(DEMOUT==0))[None,:]
    XM_T = XM_mat/(MOUT+(MOUT==0))[None,:]

    MD_T = MD_mat/(DOUT+(DOUT==0))[None,:]
    MDEM_T = MDEM_mat/(DEMOUT+(DEMOUT==0))[None,:]

    FD_T = FD_mat/(DOUT+(DOUT==0))[None,:]

    if not os.path.isdir(os.path.abspath('scripts/data/computed/blocks')):
        os.mkdir(os.path.abspath('scripts/data/computed/blocks'))
    if not os.path.isdir(os.path.abspath('scripts/data/computed/blocks/industry')):
        os.mkdir(os.path.abspath('scripts/data/computed/blocks/industry'))
    
    blocks = [
        [DD_T,                     XM_T,                     np.zeros(FD_T.T.shape),                DDEM_T,                   ],
        [MD_T,                     np.zeros(DD_T.shape),     np.zeros(FD_T.T.shape),                MDEM_T,                   ],
        [FD_T,                     np.zeros(FD_T.shape),     np.zeros([num_reg*num_fac]*2),         np.zeros([num_reg*num_fac,num_reg*3]), ],
        [np.zeros(DDEM_T.T.shape), np.zeros(DDEM_T.T.shape), np.zeros([num_reg*3,num_reg*num_fac]), np.zeros([num_reg*3]*2),  ],
    ]
    for i in range(4):
        for k in range(4):
            np.save('scripts/data/computed/blocks/industry/'+names[i]+names[k]+'_T.npy',blocks[i][k])