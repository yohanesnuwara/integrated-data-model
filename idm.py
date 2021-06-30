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

def visualize_trajectory(df, x, y, z):
  fig = px.line_3d(df, x=x, y=y, z=z)
  fig.update_scenes(zaxis_autorange="reversed")
  fig.show()

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
