import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
df=pd.read_csv("train.csv")
customersdata = pd.read_csv("train.csv")
st.markdown("<h1 style='text-align: center; color: red;'>Customer Segmenation Analysis</h1>", unsafe_allow_html=True)

st.dataframe(df)
df['Spending_Score']= df['Spending_Score'].replace('Low',0)
df['Spending_Score']= df['Spending_Score'].replace('High',2)
df['Spending_Score']= df['Spending_Score'].replace('Average',1)

customersdata=df

df['Gender']= df['Gender'].replace('Male',0)
df['Gender']= df['Gender'].replace('Female',1)
df.fillna(0)
customersdata=df
customersdata=customersdata.fillna(0)

kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)

# Train the model
kmeans_model.fit(customersdata[['Spending_Score','Family_Size',
'Gender']])

# Create the K means model for different values of K
def try_different_clusters(K, data):

    cluster_values = list(range(1, K+1))
    inertias=[]

    for c in cluster_values:
        model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    return inertias

# Find output for k values between 1 to 12 
outputs = try_different_clusters(12, customersdata[['Spending_Score','Family_Size','Gender']])
distances = pd.DataFrame({"clusters": list(range(1, 13)),"sum of squared distances": outputs})

figure = go.Figure()
figure.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

figure.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),
                  xaxis_title="Number of clusters",
                  yaxis_title="Sum of squared distances",
                  title_text="Finding optimal number of clusters using elbow method")

st.plotly_chart(figure, use_container_width=True)


# Re-Train K means model with k=5
kmeans_model_new = KMeans(n_clusters = 5,init='k-means++',max_iter=400,random_state=42)

st.write(kmeans_model_new.fit_predict(customersdata[['Spending_Score','Family_Size','Gender']]))

# Create data arrays
cluster_centers = kmeans_model_new.cluster_centers_
data = np.expm1(cluster_centers)
points = np.append(data, cluster_centers, axis=1)
st.write(points)

# Add "clusters" to customers data
points = np.append(points, [[0], [1], [2], [3], [4]], axis=1)
customersdata["clusters"] = kmeans_model_new.labels_

# visualize clusters
figure1 = px.scatter_3d(customersdata,
                    color='clusters',
                    x="Spending_Score",
                    y="Family_Size",
                    z="Gender",
                    category_orders = {"clusters": ["0", "1", "2", "3", "4"]}
                    )
figure1.update_layout()
# figure1.show()
st.plotly_chart(figure1, use_container_width=True)
