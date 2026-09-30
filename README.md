Data and code used for Otiniano et al., (submitted to Climate of the Past). 

1. **Data**
 - All data is stored in "~/Data/"
 - Raw proxy time series data are stored in "~/Data/Raw/"
 - Processed data is stored in "~/Data/Processed/"
2. **Analysis** 
 - Raw data is read, time binned, and prepared for FA using "~/Analysis/import data/ import_data.py"
 - FA is run for proxy time series and climate reconstruction products using "~/Analysis/FA/1. Otiniano FA.py"
  - all helper functions in "/Analysis/FA/factor_analysis"
 - Procrustes comparison is run using "~/Analysis/FA/2. Procrustes Comparison.py"
 - Code to apply FA to Briner et al., (2016) is stored in the figure "~/Figures/Figure S1/Fig S1.py"
3. **Figures** 
 - All figure generation code is stored in "~/Analysis/Figures/"
 - Raw and time-binned proxy time series figures are stored in "~/Data/Proxy Time Series/Time Series Figures/"
 - Footprints of the climate reconstruction products are illustrated in "~/Data/Climate Reconstruction Products/Footprints/"

Code tested with Python 3.11.14
install required packages from the repository's main folder:
```sh
python -m pip install -r requirements.txt```
