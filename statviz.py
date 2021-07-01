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

def visualize_trajectory(df, x, y, z):
  fig = px.line_3d(df, x=x, y=y, z=z)
  fig.update_scenes(zaxis_autorange="reversed")
  fig.show()


def summary_stats_features(df, feature):
  """ Summary statistics per feature """
  df1 = df[feature]
  # Rounded to 2 decimal places
  return df1.describe().round(2).transpose()

def summary_stats_datetime(df, feature, column_time, freq="1D"):
  """ 
  Summary statistics of a given feature per date intervals 
  (hourly, daily, monthly)
  """
  df1 = df[feature + [column_time]]
  return df1.groupby(pd.Grouper(key=column_time, freq=freq)).describe()  

def summary_stats_depth(df, feature, column_depth, interval):
  """ Summary statistics of a given feature per depth interval """
  df1 = df[feature]  
  mind, maxd, inc = min(df[column_depth]), max(df[column_depth])+interval, interval
  return df1.groupby(pd.cut(df[column_depth], np.arange(mind, maxd, inc))).describe()

def plot_timeseries(df, features, column_time):
  @interact

  def f(y_axis=features, color_by=features):
    x_axis = df[column_time].values
    cont_color = ["blue", "green", "red", "yellow"]  
    fig = px.scatter(df, x=column_time, y=y_axis, color=color_by,
                    range_x=(min(x_axis), max(x_axis)),
                    color_continuous_scale=cont_color,                   
                    width=1100, height=500)
    fig.show()
    
def boxplot_depth(df, feature, column_depth, interval):
  """ 
  Boxplot of a given feature per depth interval 
  (Plotly Go "Interactive" style)
  """
  # Ignore copy warning
  import warnings
  from pandas.core.common import SettingWithCopyWarning
  warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

  # Group dataframe by depth intervals
  df1 = df[feature]  
  mind, maxd, inc = min(df[column_depth]), max(df[column_depth])+interval, interval
  x = df1.groupby(pd.cut(df[column_depth], np.arange(mind, maxd, inc)))

  # Get the depth groups
  depth_groups = pd.DataFrame(x).iloc[:,0].astype(str).values  

  df_per_group, keys = [], []
  fig = go.Figure() # boxplot using the Plotly Go method, instead of Express

  for key, item in x:
    y = x.get_group(key)
    y["Interval"] = key
    c = np.str(key)

    fig.add_trace(go.Box(y=y[feature].values[:,0], name=c)) # Plotly Go
    fig.update_layout(xaxis_visible=False,
                      title="Boxplot of {} for {} m depth intervals".format(feature[0], interval))

    df_per_group.append(y)
    keys.append(key)

  fig.show()
  
  # Concatenate all dataframes
  df_per_group = pd.concat(df_per_group).reset_index(drop=True)
  df_per_group["Interval"] = df_per_group["Interval"].astype(str)

def boxplot_datetime(df, feature, column_time, freq="1D"):
  """ 
  Boxplot of a given feature per time interval 
  (Plotly Go "Interactive" style)
  """
  # Ignore copy warning
  import warnings
  from pandas.core.common import SettingWithCopyWarning
  warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

  # Group dataframe by depth intervals
  df1 = df[feature + [column_time]]
  x = df1.groupby(pd.Grouper(key=column_time, freq=freq))

  # Get the depth groups
  depth_groups = pd.DataFrame(x).iloc[:,0].astype(str).values  

  df_per_group, keys = [], []
  fig = go.Figure() # boxplot using the Plotly Go method, instead of Express

  for key, item in x:
    y = x.get_group(key)
    y["Interval"] = key
    c = np.str(key)

    fig.add_trace(go.Box(y=y[feature].values[:,0], name=c)) # Plotly Go
    fig.update_layout(xaxis_visible=False,
                      title="Boxplot of {} for {} time intervals".format(feature[0], freq))

    df_per_group.append(y)
    keys.append(key)

  fig.show()
  
  # Concatenate all dataframes
  df_per_group = pd.concat(df_per_group).reset_index(drop=True)
  df_per_group["Interval"] = df_per_group["Interval"].astype(str)

def kde1d(df, features, nrows=3, ncols=3, figsize=(20,10)):
  plt.figure(figsize=figsize)
  for i in range(len(features)):
    plt.subplot(nrows,ncols,i+1)
    sns.kdeplot(df[features[i]])
    plt.title(features[i])
  plt.tight_layout(1.1)
  plt.show()

def kde1d_depth(df, feature, column_depth, interval, nrows=3, ncols=3, figsize=(20,10)):
  """ KDE 1D distribution of a given feature per depth interval """
  # Ignore "Dataset has 0 variance" warning
  import warnings
  warnings.simplefilter("ignore", UserWarning)

  df1 = df[feature]  
  mind, maxd, inc = min(df[column_depth]), max(df[column_depth])+interval, interval

  # Group dataframe by depth intervals
  x = df1.groupby(pd.cut(df[column_depth], np.arange(mind, maxd, inc)))

  # Get the depth groups
  depth_groups = pd.DataFrame(x).iloc[:,0].astype(str).values

  plt.figure(figsize=figsize)
  for i in range(len(feature)):
    plt.subplot(nrows, ncols, i+1)
    plt.title(feature[i])
    # Groups of each feature
    for key, item in x:
      y = x.get_group(key)
      values = y[feature[i]].values 
      sns.kdeplot(values, Label=key)  
      plt.legend(loc="best")

  plt.tight_layout(1.1)    
  # plt.show()

def scatter2d(df, features):
  @interact

  def f(x_axis=features, y_axis=features, color_by=features):
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                    marginal_x="histogram", marginal_y="histogram",
                    width=700, height=700)
    fig.show()  
    
def scatter3d(df, features):
  @interact

  def f(x_axis=features, y_axis=features, z_axis=features):
    fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=z_axis)
    fig.show()

def kde2d(df, x_axis, y_axis, evaluate_proba=False, xbound=None, ybound=None):
  sns.kdeplot(data=df, x=x_axis, y=y_axis, fill=True, cbar=True, cmap="plasma")
  if evaluate_proba==True:
    plt.vlines(xbound[0], ymin=ybound[0], ymax=ybound[1], color="red")
    plt.vlines(xbound[1], ymin=ybound[0], ymax=ybound[1], color="red")
    plt.hlines(ybound[0], xmin=xbound[0], xmax=xbound[1], color="red")
    plt.hlines(ybound[1], xmin=xbound[0], xmax=xbound[1], color="red")
  plt.title("2D Kernel Density Estimation Plot", pad=20)
  plt.grid()

def compute_proba_1d(df, x, xrange):
  from scipy import stats

  ## Estimate KDE 1D
  kdex = stats.gaussian_kde(df[x])

  ## Integrate 
  Px = kdex.integrate_box(xrange[0], xrange[1])

  return Px

def compute_proba_2d(df, x, y, xrange, yrange):
  from scipy import stats

  # Compute marginal probabilities
  Px = compute_proba_1d(df, x, (xrange[0], xrange[1]))
  Py = compute_proba_1d(df, y, (yrange[0], yrange[1]))

  # Compute joint probabilities
  xmin, xmax = df[x].min(), df[x].max()
  ymin, ymax = df[y].min(), df[y].max()

  ## Estimate KDE 2D, P(xny)
  X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
  positions = np.vstack([X.ravel(), Y.ravel()])
  values = np.vstack([df[x], df[y]])
  kdexy = stats.gaussian_kde(values)  

  ## Integrate
  Pxny = kdexy.integrate_box([xrange[0], yrange[0]], [xrange[1], yrange[1]])

  # Compute conditional probability, or Probability of y given x occured, P(y|x)
  Pxy_cond = Pxny / Px

  return Px, Py, Pxny, Pxy_cond
