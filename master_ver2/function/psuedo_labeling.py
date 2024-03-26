import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import sys

# Plotly와 연동
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from scipy.interpolate import griddata

sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import logging_time

def merge_total_vibration_control_performance(total_vibration_performance):
    '''
    Args:                                                       DataType                    Examples:                                                                                              Description(Usage):
        total_vibration_performance                             pd.DataFrame()              (index)        I.R_0050_D.R_0067_F.P_0010_S.P_0200_I.P_0200_A.T_0.010_R.I_0003                         Extract SNR value of a Condition Variable
                                                                                            (cut_off_freq) 121.04
                                                                                            (SNR)          31.71
                                                                                            (SNR_Label)    0~6                                                                                      Extract Condition Varaible                                             
        
    Returns:
        total_performance                                       pd.DataFrame()              (index)        I.R_0050_D.R_0067_F.P_0010_S.P_0200_I.P_0200_A.T_0.010_R.I_0003                         Extract SNR value of a Condition Variable
                                                                                            (cut_off_freq) 121.04
                                                                                            (SNR)          31.71
                                                                                            (SNR_Label)    0~6
                                                                                            
    '''


    def generate_total_vibration_psuedo_labels(total_vibration_performance):
        '''
        Return the total_vibration_performance with the pseudo labels
        '''
        Methodology = 'Hierarchical'
        KMeans_Params = {'n_clusters': 5, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300, 'tol': 0.0001, 'algorithm': 'lloyd'}
        colors = dict()

        kmeans = KMeans(**KMeans_Params)
        clusters = kmeans.fit_predict(total_vibration_performance['SNR'].values.reshape(-1,1))
        # Perform KMeans Clustering by setting a larger weight the closer it is to the origin
        # ReOrder the clusters along the SNR values
        centroid = kmeans.cluster_centers_
        ord_idx=np.argsort(centroid.flatten())
        reorderd_clusters = np.zeros_like(clusters)-1
        for i in np.arange(np.unique(clusters).shape[0]):
            reorderd_clusters[clusters==ord_idx[i]]=i

        clusters = reorderd_clusters
        
        # Hierarchical Clustering
        if Methodology == 'Hierarchical':
            Additional_KMeans_Params = {'n_clusters': 3, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300, 'tol': 0.0001, 'algorithm': 'lloyd'}
            additional_kmeans = KMeans(**Additional_KMeans_Params)
            additional_clusters = additional_kmeans.fit_predict(total_vibration_performance['SNR'][clusters == 0].values.reshape(-1,1))

            # ReOrder the clusters along the SNR values
            centroid = additional_kmeans.cluster_centers_
            ord_idx=np.argsort(centroid.flatten())
            reorderd_clusters = np.zeros_like(additional_clusters)-1
            for i in np.arange(np.unique(additional_clusters).shape[0]):
                reorderd_clusters[additional_clusters==ord_idx[i]]= i + 1 - Additional_KMeans_Params['n_clusters']

            clusters[clusters==0] = reorderd_clusters

        total_vibration_performance[Methodology+'_Cluster'] = clusters
        
        return total_vibration_performance


    total_vibration_performance = generate_total_vibration_psuedo_labels(total_vibration_performance)
    
    # 1. Rename the index of vibration_performance and add the matched columns of Control Variables
    '''
        columns :  Inertia_Ratio, Dumper_Ratio, First_Pole, Second_Pole, Integral_Pole, Acc_Time, Real_Inertia
        values  :  0050, 0067, 0010, 0200, 0200, 0.010, 0003
        to columns : idx
           values : I.R_0050_D.R_0067_F.P_0010_S.P_0200_I.P_0200_A.T_0.010_R.I_0003 
    '''

    return total_vibration_performance