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
import wellpathpy as wp
import lasio

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
