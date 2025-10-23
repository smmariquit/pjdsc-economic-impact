with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write('''# Core ML and Data Science
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.5.0
joblib==1.4.2

# Streamlit and Visualization
streamlit==1.28.1
plotly==5.17.0
folium==0.15.0
streamlit-folium==0.15.0
matplotlib==3.8.2

# Geographic and Spatial
geopandas==0.14.1

# HTTP Requests
requests==2.31.0
''')
print("requirements.txt updated with matplotlib!")
