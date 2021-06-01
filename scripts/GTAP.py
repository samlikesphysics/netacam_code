import numpy as np
from stoclust.Group import Group
from stoclust.Hierarchy import Hierarchy
import geopandas

class GTAP:
    '''
    A class which is useful for interfacing
    with the GTAP data set.

    The sectors, regions, commodities and factors
    of GTAP 8 are stored as Groups within Hierarchies,
    which are classes from the stoclust
    package. These Hierarchies allow one to
    easily aggregate variables into natural categories.

    The fetch_sam, fetch_carbon, fetch_labor,
    and similar functions allow for the interfacing to
    GTAP's satellite data in a manner that naturally incorporates
    Aggregations and Hierarchies.

    The geodata method delivers a Pandas GeoDataFrame
    which can be used for making Choropleths.
    '''
    def __init__(self):
        sectors = Group(np.load('scripts/data/ID/sectors.npy'))
        regions = Group(np.load('scripts/data/ID/regions.npy'))
        commodities = Group(np.load('scripts/data/ID/commodities.npy'))
        factors = Group(np.load('scripts/data/ID/factors.npy'),superset=sectors)

        # Creating the sector hierarchy
        imports = Group(['m_'+c for c in commodities], superset=sectors)
        activities = Group(['a_'+c for c in commodities], superset=sectors)
        domestic = Group(['d_'+c for c in commodities], superset=sectors)

        tariffs = Group(['tmm_'+r for r in regions]+['tee_'+r for r in regions],superset=sectors)
        duties = Group(['tssm_'+c for c in commodities]+['tssd_'+c for c in commodities],superset=sectors)
        factor_taxes = Group(['tf_'+f for f in factors],superset=sectors)

        land_margins = Group(['otp_'+r for r in regions],superset=sectors)
        water_margins = Group(['wtp_'+r for r in regions],superset=sectors)
        air_margins = Group(['atp_'+r for r in regions],superset=sectors)
        margin_incomes = Group(['otp_pvst','wtp_pvst','atp_pvst'], superset=sectors)

        trade = Group(['ww_'+r for r in regions],superset=sectors)

        consumers = Group(['PRIV','GOVT'],superset=sectors)

        self.sectors = Hierarchy(
            sectors,
            Group(
                list(sectors)+
                [
                    'imports',
                    'domestic',
                    'activities',
                    'tariffs',
                    'duties',
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
                    'final_demand',
                ],
            ),
            {
                sectors.size+0 : (1,imports.in_superset),
                sectors.size+1 : (1,domestic.in_superset),
                sectors.size+2 : (1,activities.in_superset),
                sectors.size+3 : (1,tariffs.in_superset),
                sectors.size+4 : (1,duties.in_superset),
                sectors.size+5 : (1,factor_taxes.in_superset),
                sectors.size+6 : (1,land_margins.in_superset),
                sectors.size+7 : (1,water_margins.in_superset),
                sectors.size+8 : (1,air_margins.in_superset),
                sectors.size+9 : (1,trade.in_superset),
                sectors.size+10 : (1,consumers.in_superset),
                sectors.size+11 : (2,margin_incomes.in_superset),
                sectors.size+12 : (2,np.array([
                                            sectors.size+6,
                                            sectors.size+7,
                                            sectors.size+8
                                        ])),
                sectors.size+13 : (3,np.array([
                                            sectors.size+0,
                                            sectors.size+1,
                                            sectors.size+2
                                        ])),
                sectors.size+14 : (3,factors.in_superset),
                sectors.size+15 : (3,np.array([
                                            sectors.size+3,
                                            sectors.size+4,
                                            sectors.size+5,
                                            sectors.ind['PRODTAX'],
                                            sectors.ind['DIRTAX']
                                        ])),
                sectors.size+16 : (3,np.array([
                                            sectors.size+11,
                                            sectors.size+12
                                        ])),
                sectors.size+17 : (3,np.array([
                                            sectors.size+10,
                                            sectors.ind['CGDS']
                                        ])),
            }
        )

        # Creating the commodity hierarchy
        vegetable = Group(['pdr', 'wht', 'gro', 'v_f', 'osd', 'c_b', 'pfb', 'ocr'],superset=commodities)
        animal = Group(['ctl', 'oap', 'rmk', 'wol','fsh'],superset=commodities)
        food_prod = Group(['cmt','omt','vol','mil','pcr','sgr','ofd','b_t'],superset=commodities)

        energy_extraction = Group(['coa','oil','gas'],superset=commodities)

        light_manufacture = Group(['tex','wap','lea','lum','ppp'],superset=commodities)

        chemical_manufacture = Group(['p_c','crp',],superset=commodities)
        material_manufacture = Group(['nmm','nfm'],superset=commodities)
        product_manufacture = Group(['mvh','otn','ele','ome','omf','fmp'],superset=commodities)

        utility = Group(['ely','gdt','wtr'],superset=commodities)
        margins = Group(['otp','wtp','atp'],superset=commodities)
        business = Group(['ofi','isr','obs'],superset=commodities)
        public = Group(['cmn','ros','osg'],superset=commodities)

        new_names = list(commodities)
        new_names[commodities.ind['i_s']] = 'Iron/Steel'
        new_names[commodities.ind['dwe']] = 'Dwellings'
        new_names[commodities.ind['cns']] = 'Construction'
        new_names[commodities.ind['trd']] = 'Trade and Retail'
        new_names[commodities.ind['omn']] = 'Mineral Extraction'

        self.commodities = Hierarchy(
            commodities,
            Group(
                list(commodities)+
                [
                    'Plant Harvest',
                    'Animal Harvest',
                    'Food Product',
                    'Light Material',
                    'Manufactured Chemical',
                    'Non-Ferrous Material',
                    'Manufactured Product',
                    'Utility Distribution',
                    'Transportation',
                    'Business Service',
                    'Public Service',
                    'All Land Use',
                    'Energy Extraction',
                    'All Extraction',
                    'All Manufacturing',
                    'All Services',
                ],
            ),
            {
                commodities.size+0 : (1,vegetable.in_superset),
                commodities.size+1 : (1,animal.in_superset),
                commodities.size+2 : (1,food_prod.in_superset),
                commodities.size+3 : (1,light_manufacture.in_superset),
                commodities.size+4 : (1,chemical_manufacture.in_superset),
                commodities.size+5 : (1,material_manufacture.in_superset),
                commodities.size+6 : (1,product_manufacture.in_superset),
                commodities.size+7 : (1,utility.in_superset),
                commodities.size+8 : (1,margins.in_superset),
                commodities.size+9 : (1,business.in_superset),
                commodities.size+10 : (1,public.in_superset),
                commodities.size+11 : (2,np.array([
                                            commodities.size+0,
                                            commodities.size+1,
                                            commodities.ind['frs']
                                        ])),
                commodities.size+12 : (1,energy_extraction.in_superset),
                commodities.size+13 : (3,np.array([
                                            commodities.size+12,
                                            commodities.ind['omn']
                                        ])),
                commodities.size+14 : (3,np.array([
                                            commodities.size+2,
                                            commodities.size+3,
                                            commodities.size+4,
                                            commodities.size+5,
                                            commodities.size+6,
                                        ])),
                commodities.size+15 : (3,np.array([
                                            commodities.size+8,
                                            commodities.size+9,
                                            commodities.size+10,
                                            commodities.ind['trd'],
                                        ])),
            }
        )

        self.commodities.clusters = Group(np.array(
            new_names + [
                    'Plant Harvest',
                    'Animal Harvest',
                    'Food Product',
                    'Light Material',
                    'Manufactured Chemical',
                    'Non-Ferrous Material',
                    'Manufactured Product',
                    'Utility Distribution',
                    'Transportation',
                    'Business Services',
                    'Public Services',
                    'All Land Use',
                    'Energy Extraction',
                    'All Extraction',
                    'All Manufacturing',
                    'All Services',
                ],
        ))

        energy = Group(np.load('scripts/data/ID/commodities_energy.npy'),superset=commodities)
        fuels = Group(np.load('scripts/data/ID/commodities_fuel.npy'),superset=commodities)

        energy_mat_sectors = Group(np.array(['m_'+c for c in energy]+
                                    ['d_'+c for c in energy]+
                                    ['ww_'+r for r in regions]),superset=sectors)
        e_imports = Group(['m_'+e for e in energy],superset=energy_mat_sectors)
        e_domestic = Group(['d_'+e for e in energy],superset=energy_mat_sectors)
        e_trade = Group(['ww_'+r for r in regions],superset=energy_mat_sectors)
        self.energy = Hierarchy(
            energy_mat_sectors,
            Group(
                list(energy_mat_sectors)+
                [
                    'imports',
                    'domestic',
                    'trade'
                ]
            ),
            {
                energy_mat_sectors.size+0 : (1,e_imports.in_superset),
                energy_mat_sectors.size+1 : (1,e_domestic.in_superset),
                energy_mat_sectors.size+2 : (1,e_trade.in_superset)
            }
        )

        carbon_mat_sectors = Group(np.array(['m_'+c for c in fuels]+
                                    ['d_'+c for c in fuels]),superset=sectors)
        f_imports = Group(['m_'+e for e in fuels],superset=carbon_mat_sectors)
        f_domestic = Group(['d_'+e for e in fuels],superset=carbon_mat_sectors)
        self.fuels = Hierarchy(
            carbon_mat_sectors,
            Group(
                list(carbon_mat_sectors)+
                [
                    'imports',
                    'domestic',
                ]
            ),
            {
                carbon_mat_sectors.size+0 : (1,f_imports.in_superset),
                carbon_mat_sectors.size+1 : (1,f_domestic.in_superset)
            }
        )

        mam_sectors = Group(
            ['d_'+c for c in commodities]+
            ['m_'+c for c in commodities]+
            ['x_'+c for c in commodities]+
            ['PRIV','GOVT','KDEP','CGDS']+
            ['ww_'+r for r in regions]
        )

        mam_imports = Group(['m_'+c for c in commodities],superset=mam_sectors)
        mam_domestic = Group(['d_'+c for c in commodities],superset=mam_sectors)
        mam_exports = Group(['x_'+c for c in commodities],superset=mam_sectors)
        mam_capital = Group(['KDEP','CGDS'],superset=mam_sectors)
        mam_consumption = Group(['PRIV','GOVT'],superset=mam_sectors)
        mam_regions = Group(['ww_'+r for r in regions],superset=mam_sectors)

        self.mam_sectors = Hierarchy(
            mam_sectors,
            Group(
                list(mam_sectors)+
                [
                    'domestic',
                    'imports',
                    'exports',
                    'consumers',
                    'capital',
                    'trade',
                ]
            ),
            {
                mam_sectors.size+0 : (1,mam_domestic.in_superset),
                mam_sectors.size+1 : (1,mam_imports.in_superset),
                mam_sectors.size+2 : (1,mam_exports.in_superset),
                mam_sectors.size+3 : (1,mam_consumption.in_superset),
                mam_sectors.size+4 : (1,mam_capital.in_superset),
                mam_sectors.size+5 : (1,mam_regions.in_superset),
            }
        )

        # Land cover and times
        self.covers = Group(np.load('scripts/data/ID/covers.npy'))
        self.years = Group(np.load('scripts/data/ID/years.npy'))

        # Creating the factor hierarchy
        land = Group(np.load('scripts/data/ID/factors_land.npy'),superset=factors)
        labor = Group(np.load('scripts/data/ID/factors_lab.npy'),superset=factors)

        self.factors = Hierarchy(
            factors,
            Group(
                list(factors)+
                [
                    'Land',
                    'Labor'
                ]
            ),
            {
                factors.size+0 : (1,land.in_superset),
                factors.size+1 : (1,labor.in_superset)
            }
        )

        # Creating the regional hierarchy

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
        agg = {regions.size+0:(1,np.arange(0,3)),    #Oceania
            regions.size+1:(1,np.arange(3,10)),   #East Asia
            regions.size+2:(1,np.arange(10,19)),  #Southeast Asia
            regions.size+3:(1,np.arange(19,25)),  #South Asia
            regions.size+4:(1,np.arange(25,29)),  #North America
            regions.size+5:(1,np.arange(29,40)),  #South America
            regions.size+6:(1,np.arange(40,47)),  #Central America
            regions.size+7:(1,np.arange(48,76)),  #EFTA
            regions.size+8:(1,np.arange(76,84)),  #Eastern Europe
            regions.size+9:(1,np.arange(85,88)),  #Central Asia
            regions.size+10:(1,np.arange(88,101)), #West Asia
            regions.size+11:(1,np.arange(101,105)),#North Africa
            regions.size+12:(1,np.arange(105,115)),#West Africa
            regions.size+13:(1,np.arange(117,129)),#East Africa
            regions.size+14:(1,np.arange(129,133)),#South Africa
            regions.size+15:(2,np.array([84,regions.size+7,regions.size+8])),  #Europe
            regions.size+16:(3,np.array([regions.size+0,
                                            regions.size+1,
                                            regions.size+2,
                                            regions.size+3,
                                            regions.size+4,
                                            regions.size+5,
                                            regions.size+6,
                                            47,                 #Caribbean
                                            regions.size+9,
                                            regions.size+10,
                                            regions.size+11,
                                            115,116,            #Central Africa, South Central Africa
                                            regions.size+12,
                                            regions.size+13,
                                            regions.size+14,
                                            regions.size+15,])),  #World
            }
        new_names = [_rename(regions[j],j,names) for j in range(regions.size)]
        clusters = Group(np.array(list(regions.elements)
                                  +['OCEANIA','EAST ASIA','SOUTHEAST ASIA','SOUTH ASIA',
                                    'NORTH AMERICA','SOUTH AMERICA','CENTRAL AMERICA','EFTA',
                                    'EASTERN EUROPE','CENTRAL ASIA','WEST ASIA','NORTH AFRICA',
                                    'WEST AFRICA','EAST AFRICA','SOUTHERN AFRICA','EUROPE','WORLD']))

        self.regions = Hierarchy(regions,clusters,agg)
        self.regions.clusters = Group(np.array(new_names+['OCEANIA','EAST ASIA','SOUTHEAST ASIA',
                                                         'SOUTH ASIA','NORTH AMERICA','SOUTH AMERICA',
                                                         'CENTRAL AMERICA','EFTA','EASTERN EUROPE',
                                                         'CENTRAL ASIA','WEST ASIA','NORTH AFRICA',
                                                         'WEST AFRICA','EAST AFRICA','SOUTHERN AFRICA',
                                                         'EUROPE','WORLD']))

        # Creating a custom hierarchy
        agg = {regions.size+0:(1,np.arange(0,3)),   #Oceania
            regions.size+1:(1,np.array([4,5,6,8,9,15])),#East Asia
            regions.size+2:(1,np.array([10,11,12,13,14,16,17,18])),  #Southeast Asia
            regions.size+3:(1,np.arange(19,25)),  #South Asia
            regions.size+4:(1,np.array([25,26,28])),  #Northern America
            regions.size+5:(1,np.arange(29,40)),  #South America
            regions.size+6:(1,np.array([27]+list(np.arange(40,48)))),  #Central America
            regions.size+7:(1,np.array([48,49,52,54,55,56,59,60,63,64,65,67,70,71,72,73,74,75])),  #Western Europe
            regions.size+8:(1,np.array([50,51,53,57,58,61,62,66,68,69,76,77,79,80,84])),  #Eastern Europe
            regions.size+9:(1,np.array([7,78,81,82,83,85,86,87,88,89,90])),  #Former USSR
            regions.size+10:(1,np.arange(91,101)), #West Asia
            regions.size+11:(1,np.arange(101,105)),#North Africa
            regions.size+12:(1,np.array([105,106,108,109,110,111,112,113,114])),#West Africa
            regions.size+13:(1,np.array([107,115,116])),#Middle Africa
            regions.size+14:(1,np.arange(117,129)),#East Africa
            regions.size+15:(1,np.arange(129,133)),#South Africa
            regions.size+16:(2,np.array([3,regions.size+0,
                                            regions.size+1,
                                            regions.size+2,
                                            regions.size+3,
                                            regions.size+4,
                                            regions.size+5,
                                            regions.size+6,
                                            regions.size+7,
                                            regions.size+8,
                                            regions.size+9,
                                            regions.size+10,
                                            regions.size+11,
                                            regions.size+12,
                                            regions.size+13,
                                            regions.size+14,
                                            regions.size+15,
                                            regions.size-1])),  #World
            }
        clusters = Group(np.array(list(regions.elements)
                                    +['OCEANIA','EAST ASIA','SOUTHEAST ASIA','SOUTH ASIA',
                                    'NORTHERN AMERICA','SOUTH AMERICA','C. AMERICA & CARIBBEAN','WESTERN EUROPE',
                                    'EASTERN EUROPE','FORMER USSR','WEST ASIA','NORTH AFRICA',
                                    'WEST AFRICA','MIDDLE AFRICA','EAST AFRICA','SOUTHERN AFRICA','WORLD']))

        self.megaregions = Hierarchy(regions,clusters,agg)
        self.megaregions.clusters = Group(np.array(new_names+['OCEANIA','EAST ASIA','SOUTHEAST ASIA','SOUTH ASIA',
                                    'NORTHERN AMERICA','SOUTH AMERICA','C. AMERICA & CARIBBEAN','WESTERN EUROPE',
                                    'EASTERN EUROPE','FORMER USSR','WEST ASIA','NORTH AFRICA',
                                    'WEST AFRICA', 'MIDDLE AFRICA','EAST AFRICA','SOUTHERN AFRICA','WORLD']))

        self.miscellaneous = {
            'xoc': ['ASM','COK','FJI','FSM','GUM','KIR','MHL','MNP','NCL','NIU','NRU','PLW','PNG','PYF','SLB','TKL','TON','TUV','VUT','WLF','WSM','PCN','UMI'],
            'xea': ['PRK','MAC'],
            'xse': ['BRN','MMR','TLS'],
            'xsa': ['AFG','BTN','MDV'],
            'xna': ['BMU','GRL','SPM'],
            'xsm': ['FLK','GUF','GUY','SUR','SGS'],
            'xca': ['BLZ'],
            'xcb': ['ABW','AIA','ANT','ATG','BHS','BRB','CUB','CYM','DMA','DOM','GRD','HTI','JAM','KNA','LCA','MSR','PRI','TCA','TTO','VCT','VGB','VIR'],
            'xef': ['ISL','LIE'],
            'xee': ['MDA'],
            'xer': ['AND','BIH','FRO','GIB','MCO','MKD','SMR','SRB','GGY','IMN','JEY','MNE','VAT','XK'],
            'xsu': ['TJK','TKM','UZB'],
            'xws': ['IRQ','JOR','LBN','PSE','SYR','YEM','NCY'],
            'xnf': ['DZA','LBY','ESH'],
            'xwf': ['CPV','GMB','GNB','LBR','MLI','MRT','NER','SHN','SLE'],
            'xcf': ['CAF','COG','GAB','GNQ','STP','TCD'],
            'xac': ['AGO','COD'],
            'xec': ['BDI','COM','DJI','ERI','MYT','RWA','SSD','SDN','SOM','SYC','SML'],
            'xsc': ['LSO','SWZ'],
            'xtw': ['ATA','ATF','BVT','IOT'],
        }

    def fetch_energy(self,regions=None,supply=None,demand=None):
        ERG = np.load('scripts/data/ERG.npy',mmap_mode='r')
        if regions is None:
            region_inds = np.arange(self.regions.items.size)
        else:
            region_inds = regions.in_superset

        if supply is None:
            supply_inds = np.arange(self.energy.items.size)
        else:
            supply_inds = supply.in_superset
        
        if demand is None:
            demand_inds = np.arange(self.sectors.items.size)
        else:
            demand_inds = demand.in_superset

        return ERG[np.ix_(region_inds,supply_inds,demand_inds)]
    
    def fetch_carbon(self,regions=None,supply=None,demand=None):
        CO2 = np.load('scripts/data/CO2.npy',mmap_mode='r')
        if regions is None:
            region_inds = np.arange(self.regions.items.size)
        else:
            region_inds = regions.in_superset

        if supply is None:
            supply_inds = np.arange(self.fuels.items.size)
        else:
            supply_inds = supply.in_superset
        
        if demand is None:
            demand_inds = np.arange(self.sectors.items.size)
        else:
            demand_inds = demand.in_superset

        return CO2[np.ix_(region_inds,supply_inds,demand_inds)]

    def fetch_sam(self,regions=None,supply=None,demand=None):
        SAM = np.load('scripts/data/SAM.npy',mmap_mode='r')
        if regions is None:
            region_inds = np.arange(self.regions.items.size)
        else:
            region_inds = regions.in_superset

        if supply is None:
            supply_inds = np.arange(self.sectors.items.size)
        else:
            supply_inds = supply.in_superset
        
        if demand is None:
            demand_inds = np.arange(self.sectors.items.size)
        else:
            demand_inds = demand.in_superset

        return SAM[np.ix_(region_inds,supply_inds,demand_inds)]
    
    def fetch_area(self,regions=None,supply=None,demand=None):
        AREA = np.load('scripts/data/AREA.npy',mmap_mode='r')
        if regions is None:
            region_inds = np.arange(self.regions.items.size)
        else:
            region_inds = regions.in_superset

        if supply is None:
            supply_inds = np.arange(self.factors['Land'].size)
        else:
            supply_inds = supply.in_superset
        
        if demand is None:
            demand_inds = np.arange(self.commodities['Plant Harvest'].size)
        else:
            demand_inds = demand.in_superset

        return AREA[np.ix_(region_inds,supply_inds,demand_inds)]
    
    def fetch_tons(self,regions=None,supply=None,demand=None):
        TONS = np.load('scripts/data/TONS.npy',mmap_mode='r')
        if regions is None:
            region_inds = np.arange(self.regions.items.size)
        else:
            region_inds = regions.in_superset

        if supply is None:
            supply_inds = np.arange(self.factors['Land'].size)
        else:
            supply_inds = supply.in_superset
        
        if demand is None:
            demand_inds = np.arange(self.commodities['Plant Harvest'].size)
        else:
            demand_inds = demand.in_superset

        return TONS[np.ix_(region_inds,supply_inds,demand_inds)]

    def fetch_tons(self,regions=None,supply=None,demand=None):
        TONS = np.load('scripts/data/TONS.npy',mmap_mode='r')
        if regions is None:
            region_inds = np.arange(self.regions.items.size)
        else:
            region_inds = regions.in_superset

        if supply is None:
            supply_inds = np.arange(self.factors['Land'].size)
        else:
            supply_inds = supply.in_superset
        
        if demand is None:
            demand_inds = np.arange(self.commodities['Plant Harvest'].size)
        else:
            demand_inds = demand.in_superset

        return TONS[np.ix_(region_inds,supply_inds,demand_inds)]

    def fetch_cover(self,regions=None,supply=None,demand=None):
        LCOV = np.load('scripts/data/LCOV.npy',mmap_mode='r')
        if regions is None:
            region_inds = np.arange(self.regions.items.size)
        else:
            region_inds = regions.in_superset

        if supply is None:
            supply_inds = np.arange(self.covers.size)
        else:
            supply_inds = supply.in_superset
        
        if demand is None:
            demand_inds = np.arange(self.factors['Land'].size)
        else:
            demand_inds = demand.in_superset

        return LCOV[np.ix_(region_inds,supply_inds,demand_inds)]

    def fetch_pop(self,home_regions=None,host_regions=None):
        POP = np.load('scripts/data/POP.npy',mmap_mode='r')
        if home_regions is None:
            home_inds = np.arange(self.regions.items.size)
        else:
            home_inds = home_regions.in_superset

        if host_regions is None:
            host_inds = np.arange(self.regions.items.size)
        else:
            host_inds = host_regions.in_superset

        return POP[np.ix_(host_inds,home_inds)]

    def fetch_labor(self,home_regions=None,host_regions=None):
        LABF = np.load('scripts/data/LABF.npy',mmap_mode='r')
        if home_regions is None:
            home_inds = np.arange(self.regions.items.size)
        else:
            home_inds = home_regions.in_superset

        if host_regions is None:
            host_inds = np.arange(self.regions.items.size)
        else:
            host_inds = host_regions.in_superset

        return LABF[np.ix_(host_inds,home_inds)]

    def fetch_time_series(self,times=None,commodities=None,supply_regions=None,demand_regions=None):
        TSERIES = np.load('scripts/data/TSERIES.npy',mmap_mode='r')
        if times is None:
            time_inds = np.arange(self.years.size)
        else:
            time_inds = times.in_superset

        if commodities is None:
            commod_inds = np.arange(self.commodities.items.size)
        else:
            commod_inds = commodities.in_superset

        if supply_regions is None:
            supply_inds = np.arange(self.regions.items.size)
        else:
            supply_inds = supply_regions.in_superset
        
        if demand_regions is None:
            demand_inds = np.arange(self.regions.items.size)
        else:
            demand_inds = demand_regions.in_superset

        return TSERIES[np.ix_(time_inds,commod_inds,supply_inds,demand_inds)]

  #  def fetch_sam(self):

    def geodata(self,data=None,show_regions=None):
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        world.iso_a3[world.name=='France'] = 'FRA'
        world.iso_a3[world.name=='Kosovo'] = 'XK'
        world.iso_a3[world.name=='Norway'] = 'NOR'
        world.iso_a3[world.name=='N. Cyprus'] = 'NCY'
        world.iso_a3[world.name=='Somaliland'] = 'SML'
        if show_regions is None:
            show_regions = self.regions.items.elements

        geometries = []
        took_regions = []
        if data is not None:
            took_data = []
        for r in show_regions:
            if r[0]=='x':
                georeg = world[world.iso_a3.isin(self.miscellaneous[r])]
            else:
                georeg = world[world.iso_a3 == r.upper()]
            if len(georeg)>0:
                geometries.append(georeg.unary_union)
                took_regions.append(r.upper())
                if data is not None:
                    took_data.append(data[np.where(show_regions==r)[0][0]])

        df = {'code':took_regions,'geometry':geometries}
        if data is not None:
            df.update({'Data':took_data})
        return geopandas.GeoDataFrame(df,crs=world.crs)


def _rename(old,j,names):
    if old[0]=='x':
        if old=='xcb' or old=='xcf' or old=='xac':
            return names[j]
        else:
            return 'REST OF '+names[j]
    else:
        return names[j]