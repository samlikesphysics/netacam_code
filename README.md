# Notebooks and Scripts for "Nonequilibirum Thermodynamics and Carbon Accounting Methods" 
**NOTE: This repository has been modified to correct a faulty import and to add
an internal module of the `stoclust` package from an earlier version, as recent
changes have rendered the up-to-date package incompatible with this repository.
No other code has been modified. An older version of this repository can be
found in previous commits.**

This repository contains notebooks for reproducing the graphs and numerical results
of our manuscript []. In addition, the notebooks and scripts have some flexibility that should
allow for further exploration of the concepts by any interested investigators.
We also include instructions for acquiring the GTAP 8 data [] from the source website,
found in the file `data/README.md`.

The flow of this repository goes something like

```
data + scripts ---------> data_to_py.ipynb ----> leontief_flows.ipynb ----> attribution.ipynb ---> plots_gtap.ipynb ----> plots_null.ipynb
```

Happy trails!