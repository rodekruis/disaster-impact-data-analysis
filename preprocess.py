# preprocess.py

"""
    This script is taken from: https://github.com/rodekruis/river-flood-data-analysis;
    at river-flood-data-analysis/GoogleFloodHub/src/GRRR.ipynb where it is used,
    in combination with other functions and the 'analyse' package, to go from "raw" Mali
    impact data (MergedImpactData.csv), comprised of impact events gathered from:
    - OCHA;
    - EMDAT;
    - DesInventar;
    - DRPC Mali; 
    - DGPC Mali;
    - CatNat;
    - Relief; and
    - a text-mining algorithm,
    to a csv file with impact events per administrative unit, which can be
    found in this folder under the name: impact_events_per_admin_[...].csv.
        The functions out of the analyse package are substituted, and the
    script is adapted to be able to run as a standalone script.
        The most important functions are near the bottom.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
import geopandas as gpd


def get_shape_file(file : str) -> gpd.GeoDataFrame:
    """
    Get the shape file for a country

    :param country: the country
    :return: the GeoDataFrame
    """
    try:
        return gpd.read_file(f"data/shape_files/{file}")
    except Exception as exc:
        raise Exception(f'Error reading shapefile: {exc}')
    

def fill_cercle_from_commune(d: Dict[str, List[str]], row: pd.Series) -> str:
    """ 
    Tries to fill in an empty 'Cercle' column with the corresponding
    'Cercle' value from the 'Commune' entry in the dict if there is one

    :param dict: dictionary with commune-cercle mapping
    :param row: row from df
    :return: 'Cercle' value (hopefully)
    """
    if pd.isna(row['Cercle']) or row['Cercle'].strip() == '':
        commune = row['Commune']
        if commune in d and len(d[commune]) == 1:
            return d[commune][0]
        else:
            return row['Cercle']
    else:
        return row['Cercle']


def export_dict_impact_events_to_csv(
        dict_impact_events: Dict[str, pd.DataFrame], verbose: bool = False
    ) -> None:
    """
    Export the dictionary with impact events to a csv file
    with a MultiIndex for keys and events

    :param dict_flood_events: dictionary with flood events
    """
    list_df_impact_events = []
    for admin_unit, df in dict_impact_events.items():
        df = df.copy()
        df.reset_index(inplace = True)
        df['admin_unit'] = admin_unit
        list_df_impact_events.append(df)

    df_flood_events = pd.concat(list_df_impact_events, ignore_index = True)

    print(df_flood_events) if verbose else None
    print(f'exporting {df_flood_events.shape[0]} impact events to csv')
    df_flood_events.to_csv(
        f'data/impact_events_per_admin_{df_flood_events.shape[0]}.csv',
        sep = ';', decimal = '.', index = False)
    

def map_cercle_names_to_pcodes(
        df: pd.DataFrame,
        path: str = 'mali_ALL/mli_adm_ab_shp/mli_admbnda_adm2_1m_gov_20211220.shp',
        verbose: bool = False
        ) -> Dict[str, str]:
    """
    Maps the Cercle names in the impact data to the pcodes found in the
    shape file of the administrative units and used for the flood events

    :param df: dataframe with impact data
    :param path: path to the shape file
    :param verbose: whether to print some test prints
    :return: dictionary with Cercle names mapped to pcodes
    """
                        # load shape file and normalise columns of interest
    gdf = get_shape_file(path).to_crs('EPSG:4326')
    gdf['ADM2_FR'] = gdf['ADM2_FR'].str.strip().str.upper()\
                        .str.normalize('NFKD').str.encode('ascii',errors = 'ignore')\
                            .str.decode('utf-8') 
                        # use the print statements to compare admin unit names
                        # of the shape file and of the impact data
    # print(np.sort(gdf['ADM2_FR'].unique()))
                        # print out the identifiers in the impact data which are not
                        # found in the shape file, so we can manually check and correct
    if verbose:
        print('Identifiers in impact data not found in shape file:')
        print(np.sort(np.setdiff1d(df['Cercle'].unique(), gdf['ADM2_FR'].unique())))
                        # the print statement above and this one produce:
    """
    ['ABEIBARA' 'ANDERAMBOUKANE' 'ANSONGO' 'BAFOULABE' 'BAMAKO' 'BANAMBA'
    'BANDIAGARA' 'BANKASS' 'BARAOUELI' 'BLA' 'BOUGOUNI' 'BOUREM' 'DIEMA'
    'DIOILA' 'DIRE' 'DJENNE' 'DOUENTZA' 'GAO' 'GOUNDAM' 'GOURMA-RHAROUS'
    'INEKAR' 'KADIOLO' 'KANGABA' 'KATI' 'KAYES' 'KENIEBA' 'KIDAL' 'KITA'
    'KOLOKANI' 'KOLONDIEBA' 'KORO' 'KOULIKORO' 'KOUTIALA' 'MACINA' 'MENAKA'
    'MOPTI' 'NARA' 'NIAFUNKE' 'NIONO' 'NIORO' 'SAN' 'SEGOU' 'SIKASSO'
    'TENENKOU' 'TESSALIT' 'TIDERMENE' 'TIN-ESSAKO' 'TOMBOUCTOU' 'TOMINIAN'
    'YANFOLILA' 'YELIMANE' 'YOROSSO' 'YOUWAROU']
    <class 'numpy.ndarray'>
    ['GOUMERA' 'ESSOUK' 'INTACHDAYTE' 'AGUELHOK' 'TAMKOUTAT' 'ZEGOUA' 'SONA'
    'KOURY' 'BIRAMABOUGOU' 'SANDIA' 'TOUBAKORO' 'KENEKOUN']
    """
                        # create a mapping from Cercle names to pcodes
    mapping = gdf.set_index('ADM2_FR')['ADM2_PCODE'].to_dict()
                        # some manual corrections made by searching Google Maps and GeoView
    manual_corrections = {
        'GOUMERA' : mapping.get('KAYES'),
        'ESSOUK' : mapping.get('KIDAL'),
        'INTACHDAYTE' : mapping.get('KIDAL'),
        'AGUELHOK' : mapping.get('TESSALIT'),
        'TAMKOUTAT' : mapping.get('GAO'),
        'ZEGOUA' : mapping.get('KADIOLO'),
        'SONA' : mapping.get('YOROSSO'),
        'KOURY' : mapping.get('YOROSSO'),
        'BIRAMABOUGOU' : mapping.get('SIKASSO'),
        'SANDIA' : mapping.get('KAYES'),
        'TOUBAKORO' : mapping.get('KOULIKORO'),
        'KENEKOUN' : mapping.get('KANGABA')
    }
                        # update the mapping with the manual corrections
                        # and apply the mapping to the 'Cercle' column    
    mapping.update(manual_corrections)
                        # save PCODE as the standard an NAME as subsidiary
    df['admin_unit'] = df['Cercle'].apply(lambda x: mapping.get(x, None))
    df['admin_unit_NAME'] = df['Cercle']
    print(df.head()) if verbose else None
                        # handle unmapped Cercle values
    unmapped_cercles = df[df['admin_unit'].isnull()]['Cercle'].unique()
    if unmapped_cercles.size > 0:
        raise ValueError(f'Unmapped Cercle values: {unmapped_cercles}')
    
    return df


def merge_duplicate_events(d_events: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    filter out double impact events by checking whether events have the 
    same start date, and then merge them to an event with the start date
    of the first, and end date of the last event with the same start date

    :param dict_events: dict with au codas keys and events dfs as values
    :return: same dict, but merged
    """
    d_events_merged = {}

    for admin_unit, df_events in d_events.items():
                            # bring event_ID back as column
        df_events = df_events.reset_index()
        grouped = df_events.groupby('flood_start')

        merged_rows = []
        for flood_start, group in grouped:
                            # if one event, don't merge
            if len(group == 1):
                merged_rows.append(group.iloc[0])
            else:
                            # find the latest flood end
                            # and update columns accordingly
                idx_max_end = group['flood_end'].idxmax()
                max_end_row = group.loc[idx_max_end].to_dict()
                max_end_row['flood_end'] = group['flood_end'].max()
                max_end_row['duration'] = (max_end_row['flood_end'] - \
                                           max_end_row['flood_start']).days + 1
                merged_rows.append(max_end_row)
                     
        merged_events = pd.DataFrame(merged_rows)
        merged_events.set_index('event_ID', inplace = True)
        merged_events.sort_values('flood_start', inplace = True)
        d_events_merged[admin_unit] = merged_events            

    return d_events_merged


def process_impact_data_to_events(
        df: pd.DataFrame, verbose: bool = False
    ) -> Dict[str, pd.DataFrame]:
    """
    Process the "raw" impact data to events similar
    to the events created from the flood data, with
    specifically, per administative unit:
    - start date of the event;
    - end date of the event;
    - duration of the event; and
    
    Here, especially classification into administrative units
    is problematic, as some data is missing. How this is handled
    can vary per your specific use case, and for missing rows
    information is printed when verbose is True

    :param df: dataframe with impact data
    :param verbose: whether to be verbose
    :return: dictionary with events per administrative unit
    """                  
                        # keep only (possitbly) relevant columns, of which notable:
                        # - 'Start date' and 'End date': self-explanatory
                        # - 'Cercle': administrative unit
                        # - the rest gives more location information, which might be
                        #   useful if administrative unit information is missing
    df = df[['Start Date', 'End Date', 'Région', 'Cercle', 'Commune', 'Quartier/Villages']].copy()
                        # rename 'Start date' and 'End date' to 'flood_start' and 'flood_end'
                        # and convert them to datetime objects with flag 'raise' for errors:
                        # if any dates are missing they can be reconstructed using the
                        # 'Années', 'Mois', and 'Jours' columns, but was not needed here
    df.rename(columns = {'Start Date': 'flood_start', 'End Date': 'flood_end'}, inplace = True)
    df['flood_start'] = pd.to_datetime(df['flood_start'], format = '%d/%m/%Y', errors = 'raise')
    df['flood_end'] = pd.to_datetime(df['flood_end'], format = '%d/%m/%Y', errors = 'raise')
                        # after printing out rows with missing dates with the following line:
    """
    print(df[df['flood_start'].isnull() | df['flood_end'].isnull()]) if verbose else None
    """ 
                        # row 133 was empty, so we drop it
    df.dropna(subset = ['flood_start', 'flood_end'], inplace = True)
                        # double check if no dates are missing
    if df['flood_start'].isnull().values.any() or df['flood_end'].isnull().values.any():
        raise ValueError('Missing dates in impact data')
                        # next, we normalise 'Cercle' column to no whitespace and uppercase,
                        # and also delete any accents on top of letters or dashes etc to make
                        # sure the 'Cercle' column is as clean and uniform as possible
    df['Cercle'] = df['Cercle'].str.strip().str.upper()
    df['Cercle'] = df['Cercle'].str.normalize('NFKD').str.encode('ascii', errors = 'ignore').str.decode('utf-8')
                        # solve some other specific naming ambiguities:
    df['Cercle'] = df['Cercle'].str.replace(r'\bTOMBOUCTO\b', 'TOMBOUCTOU', regex = True)
    df['Cercle'] = df['Cercle'].str.replace(r'\bGOURMA-RH\b', 'GOURMA-RHAROUS', regex = True)
    df['Cercle'] = df['Cercle'].str.replace('NIAFUNKE + MORE LOCATIONS', 'NIAFUNKE')
    df['Cercle'] = df['Cercle'].str.replace(r'\bBANDIAGAR\b', 'BANDIAGARA', regex = True)
    df['Cercle'] = df['Cercle'].str.replace('BAROUELI', 'BARAOUELI')
    df['Cercle'] = df['Cercle'].str.replace('DIOLA', 'DIOILA')
    df['Cercle'] = df['Cercle'].str.replace('NIANFUNKE', 'NIAFUNKE')
    df['Cercle'] = df['Cercle'].str.replace('ASANGO', 'ANSONGO')
    df['Cercle'] = df['Cercle'].str.replace('KOLONDIEB', 'KOLONDIEBA')

                        # to handle missing 'Cercle' information, we make a mapping/dict
                        # with as key the Commune (which is a subset of Cercle) and as
                        # value the corresponding Cercle, which can be used after to fill
                        # in missing Cercle information. We also check if there are not
                        # illogical double entries, which would need manual handling. We
                        # also change Cercle names with 'District' to NaN, so they can maybe
                        # be picked up by the Commune-Cercle mapping
    df['Cercle'] = df['Cercle'].replace('DISTRICT', np.nan)
    df['Commune'] = df['Commune'].str.strip().str.upper()
    dict_communce_cercle = df.groupby('Commune')['Cercle'].unique().to_dict()
                        # delete NaN's from the arrays itself in the dictionary;
                        # loop over all dictionary items and remove NaN's
    for commune, cercles in dict_communce_cercle.items():
        dict_communce_cercle[commune] = [cercle for cercle in cercles if isinstance(cercle, str)]
                        # check whether a commune is associated w/ a missing or multiple cercles
    for commune, cercles in dict_communce_cercle.items():
        if len(cercles) == 1 and (not isinstance(cercles[0], str) or cercles[0].strip() == ''):
            if verbose:
                print(f"Commune '{commune}' is associated with a missing Cercle")
    for commune, cercles in dict_communce_cercle.items():
        if len(cercles) > 1:
            if verbose:
                print(f"Commune '{commune}' is associated with multiple Cercles: {cercles}")
                        # which prints:
    """
    Commune 'BLA' is associated with multiple Cercles: ['SAN', 'BLA']
    Commune 'LOGO' is associated with multiple Cercles: ['YELIMANE', 'KAYES']
    Commune 'TIENFALA' is associated with multiple Cercles: ['BOUGOUNI', 'KOULIKORO']
    """
                        # so we handle these manually (after checking Google Maps etc.)
    dict_communce_cercle['BLA'] = ['SEGOU']
    dict_communce_cercle['LOGO'] = ['KAYES']
    dict_communce_cercle['TIENFALA'] = ['KOULIKORO']

                        # next, we fill in missing 'Cercle' information with the dictionary
                        # we created above, and we print out the rows with missing 'Cercle'
    df['Cercle'] = df.apply(lambda row: fill_cercle_from_commune(dict_communce_cercle, row),
                            axis = 1)
    df_missing_cercle = df[df['Cercle'].isnull() | \
        df['Cercle'].apply(lambda x: not isinstance(x, str) or x.strip() == '')]
    if verbose:         # Note: the above modifications "saved" 13 flood events
        print(f'\nImpact events without Cercle info: {df_missing_cercle.shape[0]}')
        print(f'Impact events with Cercle info: {df.shape[0] - df_missing_cercle.shape[0]}\n')
                        # Note: if after above, 'Cercle' is still missing, we *could* try
                        # and see if the 'Region' column is (coincidently?) the same as a
                        # 'Cercle' column, but we ignore this possibility for now.
                        # Next, we separate the data which has no 'Cercle' information
                        # and drop the three unnecessary remaining columns 
    df = df[~df.index.isin(df_missing_cercle.index)]
                        #! Comment this to not drop the indicated columns and also
                        #! add them to the line: df_events = df_group[[...]] column list
                        #! Don't forget the line which gets rid of the accent in Region
    df = df.drop(columns = ['Région', 'Commune', 'Quartier/Villages'])
    # df = df.rename(columns = {'Région': 'Region'})
                        # calculate the duration of each flood event and add as column,
                        # where the difference is inclusive, so we add 1 to the result
    df['duration'] = (df['flood_end'] - df['flood_start']).dt.days + 1

                        # next, we map the names of the 'Cercle' column to the admin units
                        # in the flood data, and we check if all 'Cercle' values are mapped
    df = map_cercle_names_to_pcodes(df) 
    df_missing_mapping = df[df['admin_unit'].isnull()]
    if df_missing_mapping.shape[0] > 0:
        print(f'unmapped Cercle values: {df_missing_mapping["Cercle"].unique()}')
    # df.to_csv('../data/impact_data/impact_data_Mali_tidied.csv')
                        # lastly, we group the data by 'Cercle' and create a dictionary
                        # with the 'Cercle' as key and the corresponding dataframe as value
    dict_events = {}
                        # order the data and store in dictionary
    for cercle, df_group in df.groupby('admin_unit'):
                        # ensure chronological order of the events
        df_group = df_group.sort_values(by = ['flood_start', 'flood_end']).reset_index(drop = True)
                        # give events an identifier and set as index
        df_group['event'] = df_group.index
        df_group['event_ID'] = df_group.apply(lambda x: f"{cercle}_{x['event']}",
                                              axis = 1)
        df_group.set_index('event_ID', inplace = True)
        df_events = df_group[['admin_unit_NAME', 'admin_unit',
                              'flood_start', 'flood_end', 'duration',]]
                                # 'Region', 'Commune', 'Quartier/Villages']]
        dict_events[cercle] = df_events
        
                        # merge duplicate events
    dict_events_merged = merge_duplicate_events(dict_events)
                        # sort again
    for _, value in dict_events_merged.items():
        value.sort_values(by = ['flood_start', 'flood_end'], inplace = True)

                        # export to csv and return
    export_dict_impact_events_to_csv(dict_events_merged, verbose)
    return dict_events_merged