import numpy as np
from tqdm import tqdm
from functools import reduce
from stoclust.Group import Group
from stoclust.Aggregation import Aggregation
from stoclust.Hierarchy import Hierarchy

class GTAP:
    def __init__(self):
        sectors = Group(np.load('data/ID/sectors.npy'))
        regions = Group(np.load('data/ID/regions.npy'))
        commodities = Group(np.load('data/ID/commodities.npy'))
        factors = Group(np.load('data/ID/factors.npy'),superset=sectors)

        imports = Group(['m_'+c for c in commodities], superset=sectors)
        activities = Group(['a_'+c for c in commodities], superset=sectors)
        domestic = Group(['d_'+c for c in commodities], superset=sectors)
        commodity_sectors = Group(['m_'+c for c in commodities]
                                    +['d_'+c for c in commodities]
                                    +['a_'+c for c in commodities], superset=sectors)

        regional_taxes = Group(['tmm_'+r for r in regions]+['tee_'+r for r in regions],superset=sectors)
        industry_taxes = Group(['tssm_'+c for c in commodities]+['tssd_'+c for c in commodities],superset=sectors)
        orig_factor_taxes = Group(['tf_'+f for f in orig_factors.clusters],superset=sectors)

        self.sectors = Hierarchy(
            sectors,
            Group(
                list(sectors)+
                [
                    'm_commodities',
                    'd_commodities',
                    'a_commodities',
                    'm_tariffs',
                    'x_tariffs',
                    'm_duties',
                    'd_duties',
                    'factor_taxes',
                    'land_margins',
                    'water_margins',
                    'air_margins',
                    'trade',
                    'consumers',
                    'margin_incomes',
                    'margin_payments',
                    'commodities',
                    'factors',
                    'taxes',
                    'margins',
                    ''
                ]
            )
        )