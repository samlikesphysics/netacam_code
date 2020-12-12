Some information on how to use the data.

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
    YQHF.csv        (REGIONS)
    YQHT.csv        (REGIONS)
    YQTF.csv        (REGIONS)
```