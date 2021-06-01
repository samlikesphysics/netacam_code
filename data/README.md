Some information on how to use the data.

The GTAP 8 database can be acquired freely from the website:
https://www.gtap.agecon.purdue.edu/databases/v8/

The GMig2 can be accessed as well from this site as a satellite data set.
This README also shows how to use the Land Use and Land Cover satellite data set,
though this is not necessary for our own work. If you choose
not to use this dataset then you may simply ignore any code
involving land and land covers in the notebooks.

It comes in the form of a Windows program where the user
can set an aggregation and then view two-dimensional cross sections
of the many multidimensional arrays which comprise the dataset.
These cross-sections can be extracted as CSV files.
The reader will have to manually extract each array
to use the code in this repository. The same is true for GMig2.

The variables in the GTAP dataset are sorted into hierarchical groups.
For a useful description of the GTAP 8 structure we recommend
the supplementary material of [1].
We list the group names followed by the (size) below.

```
PROD_COMMODS (58)
    TRAD_COMMODS (57)
        LAND_COMMODS (8)
        ERG_COMMODS (6)
            FUEL_COMMODS (5)
        MARG_COMMODS (3)

FACTORS (22)
    LAND_FACTORS (18)
    LABOR_FACTORS (2)

REGIONS (134)

LAND_COVERS (7)

YEAR (15)

SECTORS (1108)
```

The following is both a schematic of how to store the files
pulled and a recommendation for the order in which to pull the tables.
Recall that each table can only be pulled as a two-dimensional cross-section.
For >2 dimensional arrays, the last two dimensions listed will be those of the cross-section;
the preceding dimensions correspond to the numbering scheme in the folder.
Any arrays whose dimensions include FACTORS or LABOR_FACTORS must be sourced from
GMig2. Similarly, if using the Land Use and Land Cover dataset,
then any arrays using LAND_COMMODS, LAND_COVERS or LAND_FACTORS must be sourced
from here.

```
data/
    AREA/           (LAND_COMMODS x REGIONS x LAND_FACTORS)
        1.csv
        ...
        8.csv
    EDF/            (ERG_COMMODS x PROD_COMMODS x REGIONS)
        1.csv
        ...
        6.csv
    EIF/            (ERG_COMMODS x PROD_COMMODS x REGIONS)
        1.csv
        ...
        6.csv
    EVFA/           (FACTORS x PROD_COMMODS x REGIONS)
        1.csv
        ...
        22.csv
    EXI/            (ERG_COMMODS x REGIONS x REGIONS)
        1.csv
        ...
        6.csv
    LCOV/           (LAND_COVERS x REGIONS x LAND_FACTORS)
        1.csv
        ...
        7.csv
    MDF/            (FUEL_COMMODS x PROD_COMMODS x REGIONS)
        1.csv
        ...
        5.csv
    MIF/            (FUEL_COMMODS x PROD_COMMODS x REGIONS)
        1.csv
        ...
        5.csv
    Q/              (LABOR_FACTORS x REGIONS x REGIONS)
        1.csv
        2.csv
    REME/           (LABOR_FACTORS x REGIONS x REGIONS)
        1.csv
        2.csv   
    SAM/            (REGIONS x SECTORS x SECTORS)  
        1.csv
        ...
        134.csv
    TONS/           (LAND_COMMODS x REGIONS x LAND_FACTORS)
        1.csv
        ...
        8.csv
    VFAS/           (LABOR_FACTORS x REGIONS x REGIONS)
        1.csv
        2.csv  
    VFM/            (FACTORS x PROD_COMMODS x REGIONS)
        1.csv
        ...
        22.csv
    VOAS/           (LABOR_FACTORS x REGIONS x REGIONS)
        1.csv
        2.csv  
    VOMS/           (LABOR_FACTORS x REGIONS x REGIONS)
        1.csv
        2.csv  
    VTTS/           (YEAR x TRAD_COMMODS x REGIONS x REGIONS)
        Y1995/
            1.csv
            ...
            57.csv
        ...
        Y2009/
            1.csv
            ...
            57.csv
    AEZS.csv        (LAND_COMMODS)
    COVS.csv        (LAND_COVERS)
    EC.csv          (ERG_COMMODS)
    EDG.csv         (ERG_COMMODS x REGIONS)
    EDP.csv         (ERG_COMMODS x REGIONS)
    EIG.csv         (ERG_COMMODS x REGIONS)
    EIP.csv         (ERG_COMMODS x REGIONS)
    EVOA.csv        (FACTORS x REGIONS)
    FC.csv          (FUEL_COMMODS)
    H1.csv          (REGIONS)
    H2.csv          (TRAD_COMMODS)
    H6.csv          (FACTORS)
    MDG.csv         (FUEL_COMMODS x REGIONS)
    MDP.csv         (FUEL_COMMODS x REGIONS)
    MIG.csv         (FUEL_COMMODS x REGIONS)
    MIP.csv         (FUEL_COMMODS x REGIONS)
    MLND.csv        (LAND_COMMODS)
    POP.csv         (REGIONS x REGIONS)
    SAVE.csv        (REGIONS)
    SSET.csv        (SECTORS)
    VOM.csv         (FACTORS x REGIONS)
```

[1] "Bound by Chains of Carbon." Luke Bergmann. 2013