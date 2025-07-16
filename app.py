from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

app = Flask(__name__)

# Load dataset
DATA_PATH = r"D:\project\Mall Customer Segmentation\mall_dashboard\Mall_Customers.csv"
df = pd.read_csv(DATA_PATH)

# Preprocessing
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster summary (mean values)
summary = df.groupby('Cluster')[features + ['Gender', 'Age']].mean().round(2).reset_index()

# Scatter plot
scatter_fig = px.scatter(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color='Cluster',
    title='Customer Segments (K-Means)',
    hover_data=['Age', 'Gender']
)
scatter_html = scatter_fig.to_html(full_html=False)

# Bar plot for cluster averages
melted_summary = summary.melt(id_vars='Cluster', var_name='Feature', value_name='Average')
bar_fig = px.bar(
    melted_summary,
    x='Cluster',
    y='Average',
    color='Feature',
    barmode='group',
    title='Average Stats per Cluster'
)
bar_html = bar_fig.to_html(full_html=False)

@app.route('/')
def index():
    return render_template(
        'index.html',
        scatter_plot=scatter_html,
        bar_plot=bar_html,
        tables=[summary.to_html(classes='table table-bordered table-striped', index=False)]
    )

if __name__ == '__main__':
    app.run(debug=True)

