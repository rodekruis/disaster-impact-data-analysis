# **A specific case of merging and preprocessing impact data for flood forecasting in Mali**

This repository contains scripts to merge and preprocess impact data from several sources into uniform "flood events," i.e. events with a time, location, and other metadata, readily-usable for other analyses.
As impact data can, and usually is, messy and specific, most of the code can be used as inspiration, but will not be re-usable for other data.

The scripts are part of another [project](https://github.com/rodekruis/river-flood-data-analysis), which does an impact-based assessment and comparison of different riverine flood forecasting tools in Mali (of which notable, [GloFAS](https://global-flood.emergency.copernicus.eu/) and [Google Flood Hub](https://sites.research.google/floods/l/0/0/3)).

**Contributors**: Els Kuipers ([elskuipers@rodekruis.nl](mailto:elskuipers@rodekruis.nl)) and Tijn Oldenburg [toldenburg@rodekruis.nl](mailto:toldenburg@rodekruis.nl).

**Contact**: [valentijn7](https://github.com/valentijn7); [toldenburg@rodekruis.nl](mailto:toldenburg@rodekruis.nl)

## **Data**

The pre-merged data comes from various sources:
- OCHA;
- EMDAT;
- DesInventar;
- DRPC Mali;
- DGPC Mali;
- CatNat;
- Relief; 
- and a text-mining algorithm applied to newspaper articles between 2009 and 2017,

and is not stored in the repository, though can be provided upon request. Of the merged and resulting .csv's a sample is placed in the ``data/`` folder.

### **Input**

To run ``merge.py``, some paths to the original data sources need to be specified. ``preprocess.py`` can be tested using the sample data.

The paths/inputs for ``merge.py``:
- **paths**: specify the directory where your data is stored (i.e. ``DataDir``);
    - `MasterInnondationPath`: subpath to the master innondation (flood) data file (.csv);
    - `DesinventarTextminingPath`: subpath to the DesInventar text-mined data file (.csv).
- **inputs**: all the data/.csv's stored in the aforementioned paths.

The inputs for ``preprocess.py``:
- **inputs**: 'impact_data_Mali', a .csv with the output from ``merge.py``, containing the merged impact data, in addition to shapefiles (see ``data/shape_files``) for classification into administrative units (on a level of choice).

### Output

``merge.py`` outputs a merged .csv of impact data from aforementioned sources, of which a sample is ``data/SAMPLE_impact_data_Mali.csv``. ``preprocess.py`` preprocesses these to uniform impact events sorted and classified per administrative unit, see ``data/SAMPLE_impact_events_per_admin.csv`` for a sample.

## Codes

### Structure:

(1) `merge.py` goes through the several sources (named above), standardises some of their elements (such as DateTime information), extracts central information such as admin unit (also called 'Cercle'), limits events with "unrealistic" durations, and exports a .csv.

(2) `preprocess.py` preprocesses the merged impact data to generate consistent impact events per administrative unit. It cleans and normalises by renaming columns, standardising 'Cercle' (i.e. administrative unit names, and applying manual corrections for other inconsistencies. It handles some of the missing or invalid 'Cercle' information by mapping known 'Commune' names to their respective known 'Cercle' (Commune is a subset of Cercle). Then, 'Cercle' names are mapped to standardised PCODES using shapefiles, and manual mapping is done for then still unmatched names. Events get a duration, unique identifiers, are sorted, and finally filtered for duplicates.

### Setup

Because the impact data cannot be shared here, the code is not completely runnable. ``preprocess.py`` can be ran, however, though with only a sample of the impact data. To run for your own purposes, we'd suggest copying the functions and parts of interest into a script/notebook for experimentation, or the impact data can be provided upon request (to one of the contributors, or [pphung@redcross.nl](pphung@redcross.nl)).

#### Requirements

Python 3 and packages: `pandas`, `numpy`, `geopandas`.

#### Installation

Install the required packages using pip:

```bash
pip install pandas numpy geopandas
