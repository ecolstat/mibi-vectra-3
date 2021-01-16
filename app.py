import base64
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from PIL import Image

# General
import glob
import pathlib

# Data types
import numpy as np
# Ignore warnings from numpy about comparisons
import warnings
import pandas as pd

# Image processing / analysis
import tifffile
from tifffile import imread, imsave
import mahotas # See other Image Analysis packages
import scipy.ndimage.morphology
import scipy.ndimage as ndi
from scipy.spatial.distance import pdist, squareform
import skimage
from skimage.filters import threshold_local
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

# Statistics and machine learning
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap  #NOTE: currently using pre-release version 0.5 for densmap. See: https://twitter.com/leland_mcinnes/status/1331999177820303360
import umap.plot
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


# Plotting / visualization
import holoviews as hv
import holoviews.operation.datashader as hd
import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.callbacks import Points

import utils # local script to remove function definitions

# Ignore warnings from numpy about comparisons
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    app.run_server(debug=True)