import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
import warnings
import streamlit as st
warnings.filterwarnings("ignore")

data = pd.read_csv("german_credit_data.csv")
st.dataframe(data)
