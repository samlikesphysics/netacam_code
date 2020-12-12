import numpy as np
import scipy.linalg as la
import scipy.special as sp
from tqdm import tqdm
from numba import jit
from functools import reduce
from stoclust.Group import Group
from stoclust.Aggregation import Aggregation
from stoclust.Hierarchy import Hierarchy

import plotly.graph_objects as go
from ipywidgets import widgets
import plotly.express as px



class SAM:
    def __init__(self):
        self.sam = np.load('sam.npy')
        self.energy_mat = np.load('energy_mat.npy')
        self.carbon_mat = np.load('carbon_mat.npy')
        self.time_series = np.load('time_series.npy')

        self.times = Group(np.load('times.npy'))

        self.region_names = Group([])

        self.regions = Group(np.array([r for r in np.load('regions.npy')]))
        self.countries = Group(np.array([r for r in self.regions.elements 
                                         if ((r[0]!='x'))]),
                               superset=self.regions)

        self.commodities = Group(np.load('commodities.npy'))
        self.fuels = Group(np.load('fuels.npy'),superset=self.commodities)
        self.ergs = Group(np.load('ergs.npy'),superset=self.commodities)

        factors = (np.load('factors.npy'))
        self.sectors = Group(np.load('sectors.npy'))
        self.factors = Group(factors,superset=self.sectors)
        self.carbon_mat_sectors = Group(np.array(['m_'+c for c in self.fuels]+
                                    ['d_'+c for c in self.fuels]+
                                    ['a_'+c for c in self.fuels]+['PRIV','GOVT','CGDS']),superset=self.sectors)
        self.energy_mat_sectors = Group(np.array(['m_'+c for c in self.ergs]+
                                    ['d_'+c for c in self.ergs]+
                                    ['a_'+c for c in self.ergs]+
                                    ['ww_'+r for r in self.regions]+['PRIV','GOVT','CGDS']),superset=self.sectors)

        self.population = np.loadtxt(open("POP.csv", "rb"), delimiter=",", skiprows=1,dtype=str)[:,1].astype(float)

        self._set_sector_groups()

    def get_dind(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.domestics.in_superset][:,self.activities.in_superset]
                    r2r_list.append(A - np.diag(np.diag(A)))
                else:
                    r2r_list.append(np.zeros([self.commodities.size]*2))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.commodities.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])
    
    def get_mind(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.sam[self.regions.ind[r],self.imports.in_superset][:,self.activities.in_superset])
                else:
                    r2r_list.append(np.zeros([self.commodities.size]*2))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.commodities.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_xind(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(np.diag(self.sam[self.regions.ind[r],self.domestics.in_superset,self.sectors.ind['ww_'+r]]))
                else:
                    r2r_list.append(np.diag(self.sam[self.regions.ind[r],self.domestics.in_superset,self.sectors.ind['ww_'+s]]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.commodities.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])
    
    def get_find(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.sam[self.regions.ind[r],self.factors.in_superset][:,self.activities.in_superset])
                else:
                    r2r_list.append(np.zeros([self.factors.size,self.domestics.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.factors.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_dtax(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.taxes.in_superset][:,self.activities.in_superset]\
                        +self.sam[self.regions.ind[r],self.taxes.in_superset][:,self.domestics.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.taxes.size,self.domestics.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.taxes.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_mtax(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.taxes.in_superset][:,self.imports.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.taxes.size,self.imports.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.taxes.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_ftax(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.taxes.in_superset][:,self.factors.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.taxes.size,self.factors.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.taxes.size,self.factors.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_demandtax(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.taxes.in_superset][:,self.demands.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.taxes.size,self.demands.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.taxes.size,self.demands.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_ddemand(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.domestics.in_superset][:,self.demands.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.domestics.size,self.demands.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.domestics.size,self.demands.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_mdemand(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.imports.in_superset][:,self.demands.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.imports.size,self.demands.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.imports.size,self.demands.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_demandf(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.demands.in_superset][:,self.factors.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.demands.size,self.factors.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.demands.size,self.factors.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_marg(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            stack = np.zeros([3,self.imports.size])
            A = np.sum(self.sam[self.regions.ind[r],
                                self.air_margins.in_superset][:,self.imports.in_superset],axis=0)
            W = np.sum(self.sam[self.regions.ind[r],
                                self.water_margins.in_superset][:,self.imports.in_superset],axis=0)
            O = np.sum(self.sam[self.regions.ind[r],
                                self.land_margins.in_superset][:,self.imports.in_superset],axis=0)
            stack[self.margin_payments.ind['atp_pvst']] = A
            stack[self.margin_payments.ind['wtp_pvst']] = W
            stack[self.margin_payments.ind['otp_pvst']] = O
            r2r_list.append(stack)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,3,self.imports.size])
        return np.moveaxis(r2r_mat,[0,1],[1,0])
    
    def get_margind(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            r2r_list.append(self.sam[self.regions.ind[r],self.domestics.in_superset][:,self.margin_payments.in_superset])
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.commodities.size,self.margin_payments.size])
        return r2r_mat

    def get_margdemand(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            r2r_list.append(self.sam[self.regions.ind[r],self.demands.in_superset][:,self.margin_payments.in_superset])
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.demands.size,self.margin_payments.size])
        return r2r_mat

    def get_capitaldemand(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            r2r_list.append(np.sum(self.sam[self.regions.ind[r],self.demands.in_superset][:,self.trade.in_superset],axis=1))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.demands.size])
        return r2r_mat

    def get_demandrhh(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.demands.in_superset][:,self.sectors.ind['REGHOUS']]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.demands.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.demands.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_rhhfactor(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    A = self.sam[self.regions.ind[r],self.sectors.ind['REGHOUS'],self.factors.in_superset]
                    r2r_list.append(A)
                else:
                    r2r_list.append(np.zeros([self.factors.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.factors.size])
        return r2r_mat

    def get_derg(self):
        energy_domestics = Group(np.array(['d_'+e for e in self.ergs.elements]),superset=self.energy_mat_sectors)

        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.energy_mat[self.regions.ind[r],
                                                    energy_domestics.in_superset][:,self.activities.in_superset])
                else:
                    r2r_list.append(np.zeros([energy_domestics.size,self.commodities.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,energy_domestics.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_xerg(self):
        energy_domestics = Group(np.array(['d_'+e for e in self.ergs.elements]),superset=self.energy_mat_sectors)
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                diagonal = np.zeros([self.commodities.size])
                diagonal[self.ergs.in_superset] = self.energy_mat[self.regions.ind[r],energy_domestics.in_superset,self.sectors.ind['ww_'+s]]
                r2r_list.append(np.diag(diagonal)[self.ergs.in_superset,:])

        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,energy_domestics.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_merg(self):        
        energy_imports = Group(np.array(['m_'+e for e in self.ergs.elements]),superset=self.energy_mat_sectors)
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.energy_mat[self.regions.ind[r],energy_imports.in_superset][:,self.activities.in_superset])
                else:
                    r2r_list.append(np.zeros([energy_imports.size,self.commodities.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,energy_imports.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_dergcon(self):
        energy_domestics = Group(np.array(['d_'+e for e in self.ergs.elements]),superset=self.energy_mat_sectors)
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.energy_mat[self.regions.ind[r],
                                                    energy_domestics.in_superset][:,self.consumers.in_superset])
                else:
                    r2r_list.append(np.zeros([energy_domestics.size,self.consumers.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,energy_domestics.size,self.consumers.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_mergcon(self):        
        energy_imports = Group(np.array(['m_'+e for e in self.ergs.elements]),superset=self.energy_mat_sectors)
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.energy_mat[self.regions.ind[r],energy_imports.in_superset][:,self.consumers.in_superset])
                else:
                    r2r_list.append(np.zeros([energy_imports.size,self.consumers.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,energy_imports.size,self.consumers.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_dco2(self):
        carbon_domestics = Group(np.array(['d_'+e for e in self.fuels.elements]),superset=self.carbon_mat_sectors)

        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.carbon_mat[self.regions.ind[r],
                                                    carbon_domestics.in_superset][:,self.activities.in_superset])
                else:
                    r2r_list.append(np.zeros([carbon_domestics.size,self.commodities.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,carbon_domestics.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_mco2(self):        
        carbon_imports = Group(np.array(['m_'+e for e in self.fuels.elements]),superset=self.carbon_mat_sectors)
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.carbon_mat[self.regions.ind[r],carbon_imports.in_superset][:,self.activities.in_superset])
                else:
                    r2r_list.append(np.zeros([carbon_imports.size,self.commodities.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,carbon_imports.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_dco2con(self):
        carbon_domestics = Group(np.array(['d_'+e for e in self.fuels.elements]),superset=self.carbon_mat_sectors)

        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.carbon_mat[self.regions.ind[r],
                                                    carbon_domestics.in_superset][:,self.consumers.in_superset])
                else:
                    r2r_list.append(np.zeros([carbon_domestics.size,self.consumers.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,carbon_domestics.size,self.consumers.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_mco2con(self):        
        carbon_imports = Group(np.array(['m_'+e for e in self.fuels.elements]),superset=self.carbon_mat_sectors)
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.carbon_mat[self.regions.ind[r],carbon_imports.in_superset][:,self.consumers.in_superset])
                else:
                    r2r_list.append(np.zeros([carbon_imports.size,self.consumers.size]))
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,carbon_imports.size,self.consumers.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_ind_mat(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.sam[self.regions.ind[r],self.imports.in_superset][:,self.activities.in_superset]
                                    +self.sam[self.regions.ind[r],self.domestics.in_superset][:,self.activities.in_superset]
                                    +np.diag(self.sam[self.regions.ind[r],self.sectors.ind['ww_'+r],self.imports.in_superset]))
                else:
                    mat = np.diag(self.sam[self.regions.ind[s],self.sectors.ind['ww_'+r],self.imports.in_superset])
                    r2r_list.append(mat)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,self.commodities.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_pvst_vecs(self):
        r2p_list = []
        p2r_list = []
        p2c_list = []
        for r in tqdm(self.regions.elements):
            r2p_list.append(np.sum(self.fetch_sam(view_regions=Group(np.array([r]),superset=self.regions),
                                                       supply_sectors=self.domestics,
                                                       demand_sectors=self.margin_payments)[0],axis=1))
            p2r_list.append(np.sum(self.fetch_sam(view_regions=Group(np.array([r]),superset=self.regions),
                                                       demand_sectors=self.imports,
                                                       supply_sectors=self.margins)[0],axis=0))
            p2c_list.append(np.sum(self.fetch_sam(view_regions=Group(np.array([r]),superset=self.regions),
                                                       supply_sectors=self.CGDS,
                                                       demand_sectors=self.margin_payments)[0][0]))
        r2p_mat = np.stack(r2p_list)
        p2r_mat = np.stack(p2r_list)
        p2c = np.array(p2c_list)
        return r2p_mat, p2r_mat, p2c

    def get_prod_mat(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    supplies = self.factors+self.taxes
                    supplies.set_super(self.sectors)
                    r2r_list.append(self.fetch_sam(view_regions=Group(np.array([r]),superset=self.regions),
                                                       demand_sectors=self.activities,
                                                       supply_sectors=supplies)[0]+
                                    self.fetch_sam(view_regions=Group(np.array([r]),superset=self.regions),
                                                       demand_sectors=self.domestics,
                                                       supply_sectors=supplies)[0]+
                                    self.fetch_sam(view_regions=Group(np.array([r]),superset=self.regions),
                                                       demand_sectors=self.imports,
                                                       supply_sectors=supplies)[0])
                else:
                    mat = np.zeros([supplies.size,self.commodities.size])
                    r2r_list.append(mat)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,
                                              supplies.size,
                                              self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_cons_mat(self):
        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    demand = self.demands
                    r2r_list.append(self.fetch_sam(view_regions=Group([r],superset=self.regions),
                                                       supply_sectors=self.domestics,
                                                       demand_sectors=demand)[0]
                                    +self.fetch_sam(view_regions=Group([r],superset=self.regions),
                                                       supply_sectors=self.imports,
                                                       demand_sectors=demand)[0])
                else:
                    mat = np.zeros([self.commodities.size,3])
                    r2r_list.append(mat)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,
                                              self.commodities.size,3])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_erg_to_ind_mat(self):
        energy_domestics = Group(np.array(['d_'+e for e in self.ergs.elements]),superset=self.energy_mat_sectors)
        energy_imports = Group(np.array(['m_'+e for e in self.ergs.elements]),superset=self.sectors)

        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.fetch_erg(view_regions=Group([r],superset=self.regions),
                                                       supply_sectors=energy_domestics,
                                                       demand_sectors=self.activities)[0])
                else:
                    mat = np.zeros([energy_domestics.size,self.commodities.size])
                    mat[np.arange(energy_domestics.size),self.ergs.in_superset] += (self.fetch_erg(view_regions=Group([r],superset=self.regions),
                                                                                supply_sectors=energy_domestics,
                                                                                demand_sectors=Group(['ww_'+s],superset=self.sectors))[0,:,0])
                    r2r_list.append(mat)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,energy_domestics.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_erg_to_cons_mat(self):
        energy_domestics = Group(np.array(['d_'+e for e in self.ergs.elements]),superset=self.energy_mat_sectors)

        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    demand = self.demands
                    r2r_list.append(self.fetch_erg(view_regions=Group([r],superset=self.regions),
                                                       supply_sectors=energy_domestics,
                                                       demand_sectors=demand)[0])
                else:
                    mat = np.zeros([energy_domestics.size,3])
                    r2r_list.append(mat)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,
                                              energy_domestics.size,3])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_co2_to_ind_mat(self):
        carbon_domestics = Group(np.array(['d_'+e for e in self.fuels.elements]),superset=self.carbon_mat_sectors)
        carbon_imports = Group(np.array(['m_'+e for e in self.fuels.elements]),superset=self.sectors)

        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    r2r_list.append(self.fetch_co2(view_regions=Group(np.array([r]),superset=self.regions),
                                                       supply_sectors=carbon_domestics,
                                                       demand_sectors=self.activities)[0])
                else:
                    mat = np.zeros([carbon_domestics.size,self.commodities.size])
                    mat[np.arange(carbon_domestics.size),self.fuels.in_superset] += (self.fetch_co2(view_regions=Group([r],superset=self.regions),
                                                                                supply_sectors=carbon_domestics,
                                                                                demand_sectors=Group(['ww_'+s],superset=self.sectors))[0,:,0])
                    r2r_list.append(mat)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,carbon_domestics.size,self.commodities.size])
        return np.moveaxis(r2r_mat,[1,2],[2,1])

    def get_co2_to_cons_mat(self):
        carbon_domestics = Group(np.array(['d_'+e for e in self.fuels.elements]),superset=self.carbon_mat_sectors)

        r2r_list = []
        for r in tqdm(self.regions.elements):
            for s in self.regions.elements:
                if r==s:
                    demand = self.demands
                    r2r_list.append(self.fetch_co2(view_regions=Group(np.array([r]),superset=self.regions),
                                                       supply_sectors=carbon_domestics,
                                                       demand_sectors=demand)[0])
                else:
                    mat = np.zeros([carbon_domestics.size,3])
                    r2r_list.append(mat)
        r2r_mat = np.stack(r2r_list).reshape([self.regions.size,self.regions.size,
                                              carbon_domestics.size,3])
        return np.moveaxis(r2r_mat,[1,2],[2,1])
    
    def aggregate_sectors(self,agg):
        self.sectors = agg.clusters
        blocks = [list(agg[agg.clusters[k]].in_superset) for k in np.arange(agg.clusters.size)]
        self.sam = block_sum(block_sum(self.sam,blocks,axis=1),blocks,axis=2)
        aggregations = agg.as_dict()
        new_carbon_mat_sectors = Group([agg.clusters.elements[i] for i in range(agg.clusters.size) 
                                    if np.any(np.isin(self.carbon_mat_sectors.elements,aggregations[agg.clusters.elements[i]].elements))],
                                    superset=self.sectors)
        carbon_blocks = [[self.carbon_mat_sectors.ind[s] for s in aggregations[c] 
                          if np.all(np.isin(np.array([s]),self.carbon_mat_sectors.elements))] 
                         for c in new_carbon_mat_sectors]
        new_energy_mat_sectors = Group([agg.clusters.elements[i] for i in range(agg.clusters.size) 
                                    if np.any(np.isin(self.energy_mat_sectors.elements,aggregations[agg.clusters.elements[i]].elements))],
                                    superset=self.sectors)
        energy_blocks = [[self.energy_mat_sectors.ind[s] for s in aggregations[c] 
                          if np.all(np.isin(np.array([s]),self.energy_mat_sectors.elements))] 
                         for c in new_energy_mat_sectors]
        self.carbon_mat_sectors = new_carbon_mat_sectors
        self.energy_mat_sectors = new_energy_mat_sectors
        self.carbon_mat = block_sum(block_sum(self.carbon_mat,carbon_blocks,axis=1),blocks,axis=2)
        self.energy_mat = block_sum(block_sum(self.energy_mat,energy_blocks,axis=1),blocks,axis=2)
        
    def aggregate_regions(self,agg):
        new_regions = agg.clusters
        reg_blocks = [list(b.in_superset) for k,b in agg]
        reg_agg_sam = block_sum(self.sam,reg_blocks,axis=0)
        reg_agg_carbon = block_sum(self.carbon_mat,reg_blocks,axis=0)
        reg_agg_energy = block_sum(self.energy_mat,reg_blocks,axis=0)
        self.sam = reg_agg_sam
        self.carbon_mat = reg_agg_carbon
        self.energy_mat = reg_agg_energy
        reg_agg_time_series = block_sum(block_sum(self.time_series,reg_blocks,axis=2),reg_blocks,axis=3)
        self.time_series = reg_agg_time_series
        aggregations = agg.as_dict()

        sector_agg_clusters = np.array(['m_'+c for c in self.commodities]+['d_'+c for c in self.commodities]+['a_'+c for c in self.commodities]
                                     +list(self.factors.elements)+['tmm_'+r for r in new_regions]+['tee_'+r for r in new_regions]
                                     +['tssm_'+c for c in self.commodities]+['tssd_'+c for c in self.commodities]
                                     +['tf_'+f for f in self.factors]
                                     +reduce(lambda x,y:x+y,[['otp_'+r]+['wtp_'+r]+['atp_'+r] for r in new_regions],[])
                                     +['otp_pvst','wtp_pvst','atp_pvst']
                                     +['ww_'+r for r in new_regions]+['REGHOUS','PRIV','PRODTAX','DIRTAX','GOVT','CGDS'])
        sector_agg_group = Group(sector_agg_clusters)
        sector_agg_aggs = {sector_agg_group.ind[s]:np.array([self.sectors.ind[s]]) for s in sector_agg_clusters 
                           if (s[0:4] != 'tmm_' and s[0:4] != 'tee_' and s[0:4] !='otp_' and s[0:4] !='wtp_' and s[0:4] !='atp_' and s[0:3]!= 'ww_')}
        sector_agg_aggs.update({sector_agg_group.ind['tmm_'+r]:np.array([self.sectors.ind['tmm_'+s] for s in aggregations[r]]) for r in new_regions})
        sector_agg_aggs.update({sector_agg_group.ind['tee_'+r]:np.array([self.sectors.ind['tee_'+s] for s in aggregations[r]]) for r in new_regions})
        sector_agg_aggs.update({sector_agg_group.ind['otp_'+r]:np.array([self.sectors.ind['otp_'+s] for s in aggregations[r]]) for r in new_regions})
        sector_agg_aggs.update({sector_agg_group.ind['atp_'+r]:np.array([self.sectors.ind['atp_'+s] for s in aggregations[r]]) for r in new_regions})
        sector_agg_aggs.update({sector_agg_group.ind['wtp_'+r]:np.array([self.sectors.ind['wtp_'+s] for s in aggregations[r]]) for r in new_regions})
        sector_agg_aggs.update({sector_agg_group.ind['ww_'+r]:np.array([self.sectors.ind['ww_'+s] for s in aggregations[r]]) for r in new_regions})
        sector_agg = Aggregation(self.sectors,sector_agg_group,sector_agg_aggs)

        self.regions = new_regions
        self.aggregate_sectors(sector_agg)

        self._set_sector_groups()

    def aggregate_commodities(self,agg):
        new_commodities = agg.clusters
        commod_blocks = [list(b.in_superset) for k,b in agg]
        commod_agg_time_series = block_sum(self.time_series,commod_blocks,axis=1)
        self.time_series = commod_agg_time_series
        aggregations = agg.as_dict()

        new_fuels = Group([agg.clusters.elements[i] for i in range(agg.clusters.size) 
                                    if np.any(np.isin(self.fuels.elements,aggregations[agg.clusters.elements[i]].elements))],
                                    superset=new_commodities)
        fuel_blocks = [[self.fuels.ind[s] for s in aggregations[c] 
                          if np.all(np.isin(np.array([s]),self.fuels.elements))] 
                         for c in new_fuels]
        new_ergs = Group([agg.clusters.elements[i] for i in range(agg.clusters.size) 
                                    if np.any(np.isin(self.ergs.elements,aggregations[agg.clusters.elements[i]].elements))],
                                    superset=new_commodities)
        erg_blocks = [[self.ergs.ind[s] for s in aggregations[c] 
                          if np.all(np.isin(np.array([s]),self.ergs.elements))] 
                         for c in new_ergs]
        self.fuels = new_fuels
        self.ergs = new_ergs

        sector_agg_clusters = np.array(['m_'+c for c in new_commodities]+['d_'+c for c in new_commodities]+['a_'+c for c in new_commodities]
                                     +list(self.factors.elements)+['tmm_'+r for r in self.regions]+['tee_'+r for r in self.regions]
                                     +['tssm_'+c for c in new_commodities]+['tssd_'+c for c in new_commodities]
                                     +['tf_'+f for f in self.factors]
                                     +reduce(lambda x,y:x+y,[['otp_'+r]+['wtp_'+r]+['atp_'+r] for r in self.regions],[])
                                     +['otp_pvst','wtp_pvst','atp_pvst']
                                     +['ww_'+r for r in self.regions]+['REGHOUS','PRIV','PRODTAX','DIRTAX','GOVT','CGDS'])
        sector_agg_group = Group(sector_agg_clusters)
        sector_agg_aggs = {s:np.array([self.sectors[s]]) for s in sector_agg_clusters 
                           if (s[0:2] != 'm_' and s[0:2] != 'd_' and s[0:2] !='a_' and s[0:5] !='tssm_' and s[0:5] !='tssd_')}
        sector_agg_aggs.update({sector_agg_group.ind['m_'+r]:np.array([self.sectors['m_'+s] for s in aggregations[r]]) for r in new_commodities})
        sector_agg_aggs.update({sector_agg_group.ind['d_'+r]:np.array([self.sectors['d_'+s] for s in aggregations[r]]) for r in new_commodities})
        sector_agg_aggs.update({sector_agg_group.ind['a_'+r]:np.array([self.sectors['a_'+s] for s in aggregations[r]]) for r in new_commodities})
        sector_agg_aggs.update({sector_agg_group.ind['tssm_'+r]:np.array([self.sectors['tssm_'+s] for s in aggregations[r]]) for r in new_commodities})
        sector_agg_aggs.update({sector_agg_group.ind['tssd_'+r]:np.array([self.sectors['tssd_'+s] for s in aggregations[r]]) for r in new_commodities})
        sector_agg = Aggregation(sector_agg_group,Group(sector_agg_clusters),sector_agg_aggs)

        self.commodities = new_commodities
        self.aggregate_sectors(sector_agg)

        self._set_sector_groups()

    def _set_sector_groups(self):
        self.import_taxes = Group(['tmm_'+r for r in self.regions],superset=self.sectors)
        self.export_taxes = Group(['tee_'+r for r in self.regions],superset=self.sectors)
        self.sales_domestic_taxes = Group(['tssd_'+r for r in self.commodities],superset=self.sectors)
        self.sales_import_taxes = Group(['tssm_'+r for r in self.commodities],superset=self.sectors)
        self.factor_taxes = Group(['tf_'+f for f in self.factors],superset=self.sectors)
        self.PRODTAX = Group(['PRODTAX'],superset=self.sectors)
        self.DIRTAX = Group(['DIRTAX'],superset=self.sectors)
        self.taxes = self.import_taxes+self.export_taxes+self.sales_domestic_taxes+self.sales_import_taxes+self.factor_taxes+self.PRODTAX+self.DIRTAX
        self.taxes.set_super(self.sectors)

        self.imports = Group(['m_'+c for c in self.commodities],superset=self.sectors)
        self.domestics = Group(['d_'+c for c in self.commodities],superset=self.sectors)
        self.activities = Group(['a_'+c for c in self.commodities],superset=self.sectors)
        self.commodity_sectors = self.imports+self.domestics+self.activities
        self.commodity_sectors.set_super(self.sectors)

        self.land_margins = Group(['otp_'+r for r in self.regions],superset=self.sectors)
        self.water_margins = Group(['wtp_'+r for r in self.regions],superset=self.sectors)
        self.air_margins = Group(['atp_'+r for r in self.regions],superset=self.sectors)
        self.margin_payments = Group(['otp_pvst','wtp_pvst','atp_pvst'],superset=self.sectors)
        self.margins = self.land_margins+self.water_margins+self.air_margins+self.margin_payments
        self.margins.set_super(self.sectors)

        self.trade = Group(['ww_'+r for r in self.regions],superset=self.sectors)

        self.REGHOUS = Group(['REGHOUS'],superset=self.sectors)
        self.PRIV = Group(['PRIV'],superset=self.sectors)
        self.GOVT = Group(['GOVT'],superset=self.sectors)
        self.CGDS = Group(['CGDS'],superset=self.sectors)
        self.consumers = self.PRIV+self.GOVT
        self.consumers.set_super(self.sectors)
        self.demands = self.PRIV+self.GOVT+self.CGDS
        self.demands.set_super(self.sectors)

        non_trade_sectors_a = np.array(['m_'+c for c in self.commodities]+['d_'+c for c in self.commodities]+['a_'+c for c in self.commodities]
                                     +list(self.factors.elements)+['tmm_'+r for r in self.regions]+['tee_'+r for r in self.regions]
                                     +['tssm_'+c for c in self.commodities]+['tssd_'+c for c in self.commodities]
                                     +['tf_'+f for f in self.factors]
                                     +reduce(lambda x,y:x+y,[['otp_'+r]+['wtp_'+r]+['atp_'+r] for r in self.regions],[])
                                     +['otp_pvst','wtp_pvst','atp_pvst']
                                     +['REGHOUS','PRIV','PRODTAX','DIRTAX','GOVT','CGDS'])
        self.non_trade_sectors = Group(non_trade_sectors_a,superset=self.sectors)  

    def filter_supply(self,demand_sectors,tol=1e-6):
        suppliers = Group(np.array([s for s in self.sectors.elements 
                                        if np.sum(self.sam[:,self.sectors.ind[s],demand_sectors.in_superset])>tol]),
                              superset=self.sectors)
        return suppliers

    def filter_demand(self,supply_sectors,tol=1e-6):
        demanders = Group(np.array([s for s in self.sectors.elements 
                                        if np.sum(self.sam[:,supply_sectors.in_superset,self.sectors.ind[s]])>tol]),
                              superset=self.sectors)
        return demanders

    def fetch_sam(self, view_regions = None, demand_sectors=None, supply_sectors = None):
        rules = [view_regions,supply_sectors,demand_sectors]
        groups = []
        for R in range(3):
            if rules[R] is None:
                if R == 0:
                    groups.append(np.arange(self.regions.size))
                else:
                    groups.append(np.arange(self.sectors.size))
            elif isinstance(rules[R],Group):
                groups.append(rules[R].in_superset)

        f_mat = self.sam[groups[0]][:,groups[1]][:,:,groups[2]]
        return f_mat

    def fetch_erg(self, view_regions = None, demand_sectors=None, supply_sectors = None):
        rules = [view_regions,supply_sectors,demand_sectors]
        groups = []
        for R in range(3):
            if rules[R] is None:
                if R == 0:
                    groups.append(np.arange(self.regions.size))
                elif R == 1:
                    groups.append(np.arange(self.energy_mat_sectors.size))
                elif R == 2:
                    groups.append(np.arange(self.sectors.size))
            elif isinstance(rules[R],Group):
                groups.append(rules[R].in_superset)

        f_mat = self.energy_mat[groups[0]][:,groups[1]][:,:,groups[2]]
        return f_mat

    def fetch_co2(self, view_regions = None, demand_sectors=None, supply_sectors = None):
        rules = [view_regions,supply_sectors,demand_sectors]
        groups = []
        for R in range(3):
            if rules[R] is None:
                if R == 0:
                    groups.append(np.arange(self.regions.size))
                elif R == 1:
                    groups.append(np.arange(self.carbon_mat_sectors.size))
                elif R == 2:
                    groups.append(np.arange(self.sectors.size))
            elif isinstance(rules[R],Group):
                groups.append(rules[R].in_superset)

        f_mat = self.carbon_mat[groups[0]][:,groups[1]][:,:,groups[2]]
        return f_mat

    def gtap_megaregions(self):
        names = np.array(['Australia','New Zealand', 'OCEANIA', 
                          'China', 'Hong Kong', 'Japan', 'Korea', 'Mongolia', 'Taiwan', 'EAST ASIA', 
                          'Cambodia', 'Indonesia', 'Laos','Malaysia','Philippines','Singapore',
                          'Thailand','Vietnam','SOUTHEAST ASIA',
                          'Bangladesh','India','Nepal','Pakistan','Sri Lanka','SOUTH ASIA',
                          'Canada','United States','Mexico','NORTH AMERICA',
                          'Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador',
                          'Paraguay','Peru','Uruguay','Venezuela','SOUTH AMERICA',
                          'Costa Rica','Guatemala','Honduras',
                          'Nicaragua','Panama','El Salvador','CENTRAL AMERICA','CARIBBEAN',
                          'Austria','Belgium','Cyprus','Czech Republic','Denmark','Estonia',
                          'Finland','France','Germany','Greece','Hungary','Ireland','Italy',
                          'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland',
                          'Portugal','Slovakia','Slovenia','Spain','Sweden','Great Britain',
                          'Switzerland','Norway','EFTA',
                          'Albania','Bulgaria','Belarus','Croatia',
                          'Romania','Russia','Ukraine','EASTERN EUROPE','EUROPE',
                          'Kazakhstan','Kyrghyzstan','CENTRAL ASIA',
                          'Armenia','Azerbaijan','Georgia','Bahrain','Iran','Israel',
                          'Kuwait','Oman','Qatar','Saudi Arabia','Turkey','United Arab Emirates','WEST ASIA',
                          'Egypt','Morocco','Tunisia','NORTH AFRICA',
                          'Benin','Burkina Faso','Cameroon','Ivory Coast',
                          'Ghana','Guinea','Nigeria','Senegal','Togo','WEST AFRICA',
                          'CENTRAL AFRICA','SOUTH CENTRAL AFRICA',
                          'Ethiopia','Kenya','Madagascar','Malawi','Mauritius',
                          'Mozambique','Rwanda','Tanzania','Uganda','Zambia','Zimbabwe','EAST AFRICA',
                          'Botswana','Namibia','South Africa','SOUTHERN AFRICA','WORLD'])
        agg = {self.regions.size+0:(1,np.arange(0,3)),    #Oceania
            self.regions.size+1:(1,np.arange(3,10)),   #East Asia
            self.regions.size+2:(1,np.arange(10,19)),  #Southeast Asia
            self.regions.size+3:(1,np.arange(19,25)),  #South Asia
            self.regions.size+4:(1,np.arange(25,29)),  #North America
            self.regions.size+5:(1,np.arange(29,40)),  #South America
            self.regions.size+6:(1,np.arange(40,47)),  #Central America
            self.regions.size+7:(1,np.arange(48,76)),  #EFTA
            self.regions.size+8:(1,np.arange(76,84)),  #Eastern Europe
            self.regions.size+9:(1,np.arange(85,88)),  #Central Asia
            self.regions.size+10:(1,np.arange(88,101)), #West Asia
            self.regions.size+11:(1,np.arange(101,105)),#North Africa
            self.regions.size+12:(1,np.arange(105,115)),#West Africa
            self.regions.size+13:(1,np.arange(117,129)),#East Africa
            self.regions.size+14:(1,np.arange(129,133)),#South Africa
            self.regions.size+15:(2,np.array([84,self.regions.size+7,self.regions.size+8])),  #Europe
            self.regions.size+16:(3,np.array([self.regions.size+0,
                                            self.regions.size+1,
                                            self.regions.size+2,
                                            self.regions.size+3,
                                            self.regions.size+4,
                                            self.regions.size+5,
                                            self.regions.size+6,
                                            47,                 #Caribbean
                                            self.regions.size+9,
                                            self.regions.size+10,
                                            self.regions.size+11,
                                            115,116,            #Central Africa, South Central Africa
                                            self.regions.size+12,
                                            self.regions.size+13,
                                            self.regions.size+14,
                                            self.regions.size+15,])),  #World
            }
        new_names = [rename(self.regions[j],j,names) for j in range(self.regions.size)]
        clusters = Group(np.array(list(self.regions.elements)
                                  +['OCEANIA','EAST ASIA','SOUTHEAST ASIA','SOUTH ASIA',
                                    'NORTH AMERICA','SOUTH AMERICA','CENTRAL AMERICA','EFTA',
                                    'EASTERN EUROPE','CENTRAL ASIA','WEST ASIA','NORTH AFRICA',
                                    'WEST AFRICA','EAST AFRICA','SOUTHERN AFRICA','EUROPE','WORLD']))

        megaregions = Hierarchy(self.regions,clusters,agg)
        megaregions.clusters = Group(np.array(new_names+['OCEANIA','EAST ASIA','SOUTHEAST ASIA',
                                                         'SOUTH ASIA','NORTH AMERICA','SOUTH AMERICA',
                                                         'CENTRAL AMERICA','EFTA','EASTERN EUROPE',
                                                         'CENTRAL ASIA','WEST ASIA','NORTH AFRICA',
                                                         'WEST AFRICA','EAST AFRICA','SOUTHERN AFRICA',
                                                         'EUROPE','WORLD']))
        return names, megaregions
    def alt_megaregions(self):
        names = np.array(['Australia','New Zealand', 'OCEANIA', 
                            'China', 'Hong Kong', 'Japan', 'Korea', 'Mongolia', 'Taiwan', 'EAST ASIA', 
                            'Cambodia', 'Indonesia', 'Laos','Malaysia','Philippines','Singapore',
                            'Thailand','Vietnam','SOUTHEAST ASIA',
                            'Bangladesh','India','Nepal','Pakistan','Sri Lanka','SOUTH ASIA',
                            'Canada','United States','Mexico','NORTH AMERICA',
                            'Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador',
                            'Paraguay','Peru','Uruguay','Venezuela','SOUTH AMERICA',
                            'Costa Rica','Guatemala','Honduras',
                            'Nicaragua','Panama','El Salvador','CENTRAL AMERICA','CARIBBEAN',
                            'Austria','Belgium','Cyprus','Czech Republic','Denmark','Estonia',
                            'Finland','France','Germany','Greece','Hungary','Ireland','Italy',
                            'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland',
                            'Portugal','Slovakia','Slovenia','Spain','Sweden','Great Britain',
                            'Switzerland','Norway','EFTA',
                            'Albania','Bulgaria','Belarus','Croatia',
                            'Romania','Russia','Ukraine','EASTERN EUROPE','EUROPE',
                            'Kazakhstan','Kyrghyzstan','CENTRAL ASIA',
                            'Armenia','Azerbaijan','Georgia','Bahrain','Iran','Israel',
                            'Kuwait','Oman','Qatar','Saudi Arabia','Turkey','United Arab Emirates','WEST ASIA',
                            'Egypt','Morocco','Tunisia','NORTH AFRICA',
                            'Benin','Burkina Faso','Cameroon','Ivory Coast',
                            'Ghana','Guinea','Nigeria','Senegal','Togo','WEST AFRICA',
                            'CENTRAL AFRICA','SOUTH CENTRAL AFRICA',
                            'Ethiopia','Kenya','Madagascar','Malawi','Mauritius',
                            'Mozambique','Rwanda','Tanzania','Uganda','Zambia','Zimbabwe','EAST AFRICA',
                            'Botswana','Namibia','South Africa','SOUTHERN AFRICA','WORLD'])
        agg = {self.regions.size+0:(1,np.arange(0,3)),   #Oceania
            self.regions.size+1:(1,np.array([4,5,6,8,9,15])),#East Asia
            self.regions.size+2:(1,np.array([10,11,12,13,14,16,17,18])),  #Southeast Asia
            self.regions.size+3:(1,np.arange(19,25)),  #South Asia
            self.regions.size+4:(1,np.array([25,26,28])),  #Northern America
            self.regions.size+5:(1,np.arange(29,40)),  #South America
            self.regions.size+6:(1,np.array([27]+list(np.arange(40,48)))),  #Central America
            self.regions.size+7:(1,np.array([48,49,52,54,55,56,59,60,63,64,65,67,70,71,72,73,74,75])),  #Western Europe
            self.regions.size+8:(1,np.array([50,51,53,57,58,61,62,66,68,69,76,77,79,80,84])),  #Eastern Europe
            self.regions.size+9:(1,np.array([7,78,81,82,83,85,86,87,88,89,90])),  #Former USSR
            self.regions.size+10:(1,np.arange(91,101)), #West Asia
            self.regions.size+11:(1,np.arange(101,105)),#North Africa
            self.regions.size+12:(1,np.array([105,106,108,109,110,111,112,113,114])),#West Africa
            self.regions.size+13:(1,np.array([107,115,116])),#Middle Africa
            self.regions.size+14:(1,np.arange(117,129)),#East Africa
            self.regions.size+15:(1,np.arange(129,133)),#South Africa
            self.regions.size+16:(2,np.array([3,self.regions.size+0,
                                            self.regions.size+1,
                                            self.regions.size+2,
                                            self.regions.size+3,
                                            self.regions.size+4,
                                            self.regions.size+5,
                                            self.regions.size+6,
                                            self.regions.size+7,
                                            self.regions.size+8,
                                            self.regions.size+9,
                                            self.regions.size+10,
                                            self.regions.size+11,
                                            self.regions.size+12,
                                            self.regions.size+13,
                                            self.regions.size+14,
                                            self.regions.size+15,
                                            self.regions.size-1])),  #World
            }
        new_names = [rename(self.regions[j],j,names) for j in range(self.regions.size)]
        clusters = Group(np.array(list(self.regions.elements)
                                    +['OCEANIA','EAST ASIA','SOUTHEAST ASIA','SOUTH ASIA',
                                    'NORTHERN AMERICA','SOUTH AMERICA','C. AMERICA & CARIBBEAN','WESTERN EUROPE',
                                    'EASTERN EUROPE','FORMER USSR','WEST ASIA','NORTH AFRICA',
                                    'WEST AFRICA','MIDDLE AFRICA','EAST AFRICA','SOUTHERN AFRICA','WORLD']))

        megaregions = Hierarchy(self.regions,clusters,agg)
        megaregions.clusters = Group(np.array(new_names+['OCEANIA','EAST ASIA','SOUTHEAST ASIA','SOUTH ASIA',
                                    'NORTHERN AMERICA','SOUTH AMERICA','C. AMERICA & CARIBBEAN','WESTERN EUROPE',
                                    'EASTERN EUROPE','FORMER USSR','WEST ASIA','NORTH AFRICA',
                                    'WEST AFRICA', 'MIDDLE AFRICA','EAST AFRICA','SOUTHERN AFRICA','WORLD']))
        return names, megaregions

def block_sum(a, blocks, axis=0):
    """
    Given an array and an aggregation of its indices
    along a given axis, delivers a new array whose indices
    correspond to clusters of the old indices and whose
    entries are sums over the old values.
    """
    if axis==0:
        move_a = a.copy()
    else:
        move_a = np.moveaxis(a,[0,axis],[axis,0])
    order = reduce(lambda x,y:x+y, blocks,[])
    order_a = move_a[order]
    lengths = np.array([len(b) for b in blocks])
    slices = np.concatenate([np.array([0]),np.cumsum(lengths)[:-1]])
    red_a = np.add.reduceat(order_a,slices,axis=0)
    if axis==0:
        final_a = red_a.copy()
    else:
        final_a = np.moveaxis(red_a,[0,axis],[axis,0])
    return final_a

def Choropleth(a,locations=None,dropdown=None,
                projection=None,colorscale=None,layout = None,graph_properties=None):
    if colorscale is None:
        colorscale = px.colors.sequential.Cividis
    if projection is None:
        projection = 'natural earth'
    if dropdown is not None:
        drop_inds = tuple(list(dropdown.keys()))
        drops = [widgets.Dropdown(
                    options=dropdown[j].get('options',dropdown[j]['hier'].clusters.elements),
                    value=dropdown[j].get('value',dropdown[j].get('options',dropdown[j]['hier'].clusters.elements)[-1]),
                    description=dropdown[j].get('description',None),
                ) for j in drop_inds]
        inds = [np.arange(k) for k in a.shape]
        drop_inds = tuple(list(dropdown.keys()))
        for j in range(len(drop_inds)):
            inds[drop_inds[j]] = dropdown[drop_inds[j]]['hier'][drops[j].value].in_superset
        use_mat = np.sum(a[np.ix_(*inds)],axis=drop_inds)

        trace1 = go.Choropleth(locations=locations,
                                    z=use_mat,
                                    colorscale=colorscale
                                    )
    else:
        trace1 = go.Choropleth(locations=locations,
                                    z=a,
                                    colorscale=colorscale
                                    )

    g = go.FigureWidget(data=[trace1],
                        layout=go.Layout(
                            margin=dict(l=0, r=0, t=10, b=0),
                            geo = dict(
                                landcolor = 'lightgray',
                                showland = True,
                                showcountries = False,
                                countrycolor = 'gray',
                                countrywidth = 0.5,
                                projection = dict(
                                    type = projection
                                )
                            )
                        ))
    if graph_properties is not None:
        for k,v in graph_properties.items():
            g.data[0][k] = v
    if layout is not None:
        g.update_layout(**layout)
    
    if dropdown is not None:
        response = lambda change: widget_response(change,a,dropdown,drops,g)
        for d in drops:
            d.observe(response, names="value")

        return widgets.VBox([widgets.HBox([d]) for d in drops]+[g])
    else:
        return widgets.VBox([g])
    
def widget_response(change,mat,dropdown,drops,g):
    inds = [np.arange(k) for k in mat.shape]
    drop_inds = tuple(list(dropdown.keys()))
    for j in range(len(drop_inds)):
        inds[drop_inds[j]] = dropdown[drop_inds[j]]['hier'][drops[j].value].in_superset
    use_mat = np.sum(mat[np.ix_(*inds)],axis=drop_inds)
    with g.batch_update():
        g.data[0].z = use_mat

def rename(old,j,names):
    if old[0]=='x':
        if old=='xcb' or old=='xcf' or old=='xac':
            return names[j]
        else:
            return old
    else:
        return names[j]