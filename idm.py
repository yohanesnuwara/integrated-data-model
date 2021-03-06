from __future__ import print_function
import glob, sys, os, subprocess

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import seaborn as sns
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
# import wellpathpy as wp
# import lasio

from ipywidgets import interact, interactive, fixed, interact_manual, ToggleButtons
import ipywidgets as widgets

def setup():
  """
  Setup Integrated Data Model
  """
  def pip_install(name):
    subprocess.call(['pip', 'install', name])
  pip_install('lasio')
  pip_install('wellpathpy')

##### TRAJECTORY FILES
def open_xml_trajectory(path):
  # Reading the WITSML file
  with open(path) as file:
    data = file.read()

  # Parse the WITSML file using the Beautiful library
  soup = BeautifulSoup(data, 'xml')

  # Convert to Pandas dataframe
  data = soup.find_all('data')

  # Print all tags (column names)
  colnames = set([str(tag.name) for tag in soup.find_all()])

  # Convert into dataframe with only selected column names
  columns = ['azi', 'md', 'tvd', 'incl', 'dispNs', 'dispEw']
  df = pd.DataFrame()
  for col in columns:
      df[col] = [float(x.text) for x in soup.find_all(col)]

  return df

def trajectory_trueNE(df, md_column, inc_column, azi_column, 
                      surface_northing, surface_easting):
  import wellpathpy as wp
  
  md, inc, azi = df[md_column], df[inc_column], df[azi_column]
  
  # Calculate TVD, northing, easting, dogleg severity
  tvd, northing, easting, dls = wp.mincurve.minimum_curvature(md, inc, azi)
  
  # Calculate true northing and easting by shifting to the wellhead loc
  tvd, new_northing, new_easting = wp.location.loc_to_wellhead(tvd, northing, easting,
                                                              surface_northing, 
                                                              surface_easting)
  df["TVD_calc"] = tvd 
  df["surfNs"] = new_northing
  df["surfEw"] = new_easting
  return df

##### REALTIME DRILLING FILES
def glob_files(folder, extension):
  paths = sorted(glob.glob(os.path.join(folder, "*"+extension)))  
  return paths

def open_multiple_xml_files(folder, inspect=False, selected_logs=None):
  # Glob files
  paths = glob_files(folder, ".xml")

  # Inspect, otherwise open
  if inspect == True:
    lognames_list = []
    for i in range(len(paths)):
      # Reading the WITSML file
      with open(paths[i]) as f:
          data = f.read()
      
      # Parse the WITSML file using the Beautiful library
      soup = BeautifulSoup(data, 'xml')

      # Mnemonic list
      mne = soup.find_all('mnemonicList')
      lognames = mne[0].string.split(",") 
      lognames_list.append(lognames)

      # Print how many mnemonics/columns each dataframe has
      print("Dataframe {} has {} mnemonics".format(i+1, len(lognames)))

    all_lognames = np.concatenate(lognames_list, axis=0)
    all_lognames = np.unique(all_lognames)

    # Check the availability of each mnemonic in each dataframes
    availability = []
    for i in all_lognames:
      avail_per_df = []
      for j in lognames_list:
        avail = np.any(np.array(j) == i)
        avail_per_df.append(avail)

      availability.append(avail_per_df)
    
    # Create availability dataframe
    columns = np.arange(1,len(paths)+1).astype(str)
    avail_df = pd.DataFrame(availability, columns=columns, index=all_lognames)

    # Identify which mnemonics are available in all dataframes
    available = avail_df.all(axis=1).values
    available = all_lognames[available]
    print("\n")
    print("Mnemonics exists in ALL dataframes:")
    print(*('"{}"'.format(item) for item in available), sep=', ')    

    # Print the availability dataframe
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(avail_df)

  else:
    # Inspect=False, open the file with selected log names   
    dataframes = []
    for i in paths:
      # Reading the WITSML file
      with open(i) as f:
          data = f.read()
      
      # Parse the WITSML file using the Beautiful library
      soup = BeautifulSoup(data, 'xml')

      # Mnemonic list
      mne = soup.find_all('mnemonicList')
      lognames = mne[0].string.split(",")  

      data = soup.find_all('data')

      df = pd.DataFrame(columns=lognames,
                        data=[row.string.split(',') for row in data])

      #--- Select the subset of dataframes that has the all available mnemonics
      df = df[selected_logs]

      # Append all dataframes
      dataframes.append(df)

    # Concatenate all individual dataframes
    df = pd.concat(dataframes, axis=0).reset_index(drop=True)

    # Replace blank values with nan
    df = df.replace('', np.NaN)    
  
    return df  

def open_xml_file(path):
  # Open the WITSML file
  with open(path) as file:
      data = file.read()
  
  # Parse the file with Beautiful Soup
  soup = BeautifulSoup(data, 'xml')

  # Mnemonic list
  mne = soup.find_all('mnemonicList')
  lognames = mne[0].string.split(",") 

  # Convert to Pandas dataframe
  data = soup.find_all('data')
  df = pd.DataFrame(columns=lognames,
                    data=[row.string.split(',') for row in data])

  # Replace blank values with nan
  df = df.replace('', np.nan)   

  return df

def datetime_formatter(df, column_time, format):
  df[column_time] = pd.to_datetime(df[column_time], format=format)
  return df

def non_numeric_formatter(df):
  cols = df.columns[df.dtypes.eq('object')]
  df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
  return df  

##### LOGGING/MEASUREMENT-WHILE-DRILLING
def open_multiple_las_files(folder, inspect=False):
  import lasio  
  # Glob files
  paths = glob_files(folder, ".las")

  # Inspect, otherwise open
  if inspect == True:
    lognames_list, descr_list, unit_list = [], [], []
    for i in range(len(paths)):
      # Reading the LAS file
      data = lasio.read(paths[i])
      
      # Log names
      lognames = [data.curves[i].mnemonic for i in range(len(data.curves))]
      # descr = [data.curves[i].descr for i in range(len(data.curves))]
      # unit = [data.curves[i].unit for i in range(len(data.curves))]

      lognames_list.append(lognames)  
      # descr_list.append(descr)
      # unit_list.append(unit)  

      # Print how many mnemonics/columns each dataframe has
      print("Dataframe {} has {} mnemonics".format(i+1, len(lognames)))

    all_lognames = np.concatenate(lognames_list, axis=0)
    all_lognames = np.unique(all_lognames)

    # all_descr = np.concatenate(descr_list, axis=0)
    # all_descr = np.unique(all_descr)

    # all_unit = np.concatenate(unit_list, axis=0)
    # all_unit = np.unique(all_unit)        

    # Check the availability of each mnemonic in each dataframes
    availability = []
    for i in all_lognames:
      avail_per_df = []
      for j in lognames_list:
        avail = np.any(np.array(j) == i)
        avail_per_df.append(avail)

      availability.append(avail_per_df)
    
    # Create availability dataframe
    columns = np.arange(1,len(paths)+1).astype(str)
    avail_df = pd.DataFrame(availability, columns=columns, index=all_lognames)

    # List mnemonics
    # avail_df['Unit'] = all_unit
    # avail_df['Description'] = all_descr

    # Identify which mnemonics are available in all dataframes
    available = avail_df.all(axis=1).values
    available = all_lognames[available]
    print("\n")
    print("Mnemonics exists in ALL dataframes:")
    print(*('"{}"'.format(item) for item in available), sep=', ')    

    # Print the availability dataframe
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(avail_df)

  else:
    # Inspect=False, open the file with selected log names   
    dataframes = []
    for i in range(len(paths)):
      # Reading the LAS file
      data = lasio.read(paths[i])
      df = data.df().reset_index() 

      # Subset of dataframe
      # df = df[selected_logs]     

      # Append all dataframes
      dataframes.append(df)

    # Concatenate all individual dataframes
    df = pd.concat(dataframes, axis=0).reset_index(drop=True)

    # Replace blank values with nan
    df = df.replace('', np.NaN)    
  
    return df

def list_mnemonics(lasfile):
  """
  List all available mnemonics and descriptions. 
  Outputs are in dictionary and dataframe
  """
  mnemonic_array, descr_array = [], []
  for i in range(len(lasfile.curves[:])):
    mnemonic = lasfile.curves[i].mnemonic
    descr = lasfile.curves[i].descr
    mnemonic_array.append(mnemonic)
    descr_array.append(descr)

  # Create into dictionary
  mnemonic_dict = dict(zip(mnemonic_array, descr_array))  

  return mnemonic_dict, mnemonic_df

#### MUD LOG DATA
def retrieve_mud_mech(df):
  mud_mech_df = df[["dTim.1", "mdTop", "mdBottom", "ropAv", "ropMn", "ropMx", 
                    "wobAv", "tqAv", "rpmAv", "wtMudAv", "ecdTdAv",
                    "type", "dxcAv"]].copy()

  # Convert to Panda datetime
  mud_mech_df["dTim.1"] = pd.to_datetime(mud_mech_df["dTim.1"], format="%Y-%m-%dT%H:%M:%S.%fZ")

  # Drop Null observations
  # mud_mech_df = mud_mech_df.dropna(subset=["dTim.1", "mdTop", "mdBottom"])
  mud_mech_df = mud_mech_df.dropna()

  # Taking median of "mdTop" and "mdBottom", put into new column "mdTop"
  mud_mech_df["mdTop"] = (mud_mech_df["mdBottom"] + mud_mech_df["mdTop"]) / 2

  # Delete "dTim"
  mud_mech_df = mud_mech_df.drop(columns=["mdBottom"])

  # Change "mdTop" and "dTim.1" column name to "md" and "time"
  mud_mech_df = mud_mech_df.rename(columns={"mdTop": "md", "dTim.1": "time"})

  # Sort "md" from smallest to highest because some values are messed up 
  mud_mech_df = mud_mech_df.sort_values("md", axis=0, ascending=True).reset_index(drop=True)  
  return mud_mech_df

def retrieve_mud_chrom(df):
  mud_chrom_df = df[["dTim.2", "mdBottom.2", "methAv", "methMn", "methMx", 
                     "ethAv", "ethMn", "ethMx", "propAv", "propMn", "propMx", 
                     "ibutAv", "ibutMn", "ibutMx", "nbutAv", "nbutMn", 
                     "nbutMx", "ipentAv", "ipentMn", "ipentMx", "npentAv", 
                     "npentMn", "npentMx", "gasAv", "gasPeak", "gasBackgnd"]].copy()

  # Drop Null observations
  mud_chrom_df = mud_chrom_df.dropna(subset=["dTim.2", "mdBottom.2"]).reset_index(drop=True)

  # Convert to Panda datetime
  mud_chrom_df["dTim.2"] = pd.to_datetime(mud_chrom_df["dTim.2"], format="%Y-%m-%dT%H:%M:%S.%fZ")

  # Change "mdBottom.2" and "dTim.2" column name to "md" and "time"
  mud_chrom_df = mud_chrom_df.rename(columns={"mdBottom.2": "md", "dTim.2": "time"}) 
  return mud_chrom_df 

#### MERGING
def merge_data_interpolation(df_data, df_new, xdata, ydata, xnew, kind="cubic"):
  """ 
  Merging two data by interpolation 
  
  INPUT:

  df_data: Data source for interpolation 
  df_new: Data where its values are to be interpolated
  xdata: The x column name in df_data. This value MUST exist in BOTH dataframes 
    above. Usually it is the DEPTH. 
  ydata: The y column name in df_data. This value will be the TARGET for interp.
    Could be more than one (LIST)
  xnew: The x column name in df_new
  
  THEORY:

  f(x1, x2, x3, ..., xi) = y1, y2, y3, ..., yi
  f is the interpolation function. f is applied to a new x value to produce new y
  f(xn) = yn 

  OUTPUT:

  df: It is the df_new, but now contains the newly interpolated y values
  """
  xd = df_data[xdata].values
  yd = df_data[ydata].values
  xn = df_new[xnew].values

  # Interpolation
  for i in range(len(ydata)):
    yd_ = yd[:,i]
    f = scipy.interpolate.interp1d(xd, yd_, kind=kind, fill_value="extrapolate")
    yn = f(xn)
    # yn = np.interp(xn, xd, yd_)
    df_new[ydata[i]] = yn # add new column to df_new
  
  df_new[df_new[xnew] > max(xd)] = np.nan

  return df_new
