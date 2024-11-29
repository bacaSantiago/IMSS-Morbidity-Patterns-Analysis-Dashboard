# Import the required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import json
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, zscore
import geopandas as gpd
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from statsmodels.multivariate.manova import MANOVA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram
from kmodes.kprototypes import KPrototypes
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from factor_analyzer.rotator import Rotator
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import plotly.graph_objects as go
import os
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""_EDA_
This section contains the exploratory data analysis functions for the dashboard.
"""

# Upload the data
"""_Note_
For unzipping the covid_df file you can use the following code:
```python
import zipfile
with zipfile.ZipFile('Clean_Data/Clean_COVID19MEXICO2024.zip', 'r') as zip_ref:
    zip_ref.extractall('Clean_Data/')
"""
covid_df = pd.read_csv('Clean_Data/Clean_COVID19MEXICO2024.csv', 
                       parse_dates=['FECHA_ACTUALIZACION', 'FECHA_INGRESO', 'FECHA_SINTOMAS', 'FECHA_DEF'])
febriles_df = pd.read_csv('Clean_Data/Clean_Febriles.csv', 
                          parse_dates=['FECHA_ACTUALIZACION', 'FECHA_DIAGNOSTICO'])
dengue_df = pd.read_csv('Clean_Data/Clean_Dengue.csv', 
                        parse_dates=['FECHA_ACTUALIZACION', 'FECHA_SIGN_SINTOMAS'])
morbilidad_df = pd.read_csv('Clean_Data/Morbilidad.csv', 
                            encoding='latin1')

def covid_age_gender_distribution():
    # Map gender for interpretability
    covid_df['SEXO_LABEL'] = covid_df['SEXO'].map({1: 'Female', 2: 'Male'})
    
    # Create histogram
    fig = px.histogram(
        covid_df, 
        x="EDAD", 
        color="SEXO_LABEL", 
        title="Age and Gender Distribution", 
        barmode="overlay", 
        labels={"EDAD": "Age", "SEXO_LABEL": "Gender"},
        color_discrete_map={'Male': "#006EC1", 'Female': "#52BCEC"},
        nbins=50, 
        template="seaborn"
    )
    return fig

# Load CLASIFICACION_FINAL catalog for mapping
clasificacion_df = pd.read_excel('Data_Dicts/diccionario_datos_covid/Catalogos_COVID.xlsx', sheet_name='Cat CLASIFICACION_FINAL_COVID')
clasificacion_mapping = dict(zip(
    clasificacion_df['CLAVE'], 
    ['Confirmed by association', 'Confirmed by decease', 'Confirmed by laboratory', 'Invalid', 'Not applicable', 'Suspected', 'Negative']
))

def covid_cases_over_time():
    # Convert `Period` to string for Plotly compatibility
    covid_df['FECHA_INGRESO_YEAR_MONTH'] = covid_df['FECHA_INGRESO'].dt.to_period('M').astype(str)
    
    # Group data by classification and time
    cases_by_time = covid_df.groupby(['FECHA_INGRESO_YEAR_MONTH', 'CLASIFICACION_FINAL']).size().reset_index(name='Count')
    
    # Map classification codes to labels
    cases_by_time['CLASIFICACION_FINAL_LABEL'] = cases_by_time['CLASIFICACION_FINAL'].map(clasificacion_mapping)
    
    # Line chart for cases over time
    fig = px.line(
        cases_by_time, 
        x="FECHA_INGRESO_YEAR_MONTH", 
        y="Count", 
        color="CLASIFICACION_FINAL_LABEL", 
        title="COVID-19 Cases Over Time by Classification",
        labels={"FECHA_INGRESO_YEAR_MONTH": "Year-Month", "Count": "Cases", "CLASIFICACION_FINAL_LABEL": "Classification"},
        template="seaborn",
        color_discrete_sequence=px.colors.cyclical.Phase
    )
    return fig

# Load geojson file
with open('geojsons/mexico.json') as file:
    mexico = json.load(file)
    
# Load entidad catalog for mapping
entidades_df = pd.read_excel('Data_Dicts/diccionario_datos_covid/Catalogos_COVID.xlsx', sheet_name='Catálogo de ENTIDADES')
entidades_mapping = dict(zip(
    entidades_df['CLAVE_ENTIDAD'], 
    entidades_df['ENTIDAD_FEDERATIVA'].str.title()  # Match GeoJSON property
))

for i, name in zip([30, 16, 5, 9], ['Veracruz', 'Michoacán', 'Coahuila', 'Ciudad de México']):
    entidades_mapping[i] = name
    
def covid_cases_geospatial():
    # Map state codes to state names
    covid_df['STATE_LABEL'] = covid_df['ENTIDAD_RES'].map(entidades_mapping)
    
    # Aggregate cases by state
    cases_by_state = covid_df.groupby('STATE_LABEL').size().reset_index(name='Case_Count')
    
    # Geospatial distribution of cases
    fig = px.choropleth(
        cases_by_state, 
        locations="STATE_LABEL", 
        geojson=mexico, 
        featureidkey="properties.name", 
        color="Case_Count", 
        title="COVID-19 Cases by State",
        color_continuous_scale="Blues",
        range_color=(0, 150000),
        labels={"Case_Count": "Cases"},
        template="seaborn"
    )
    fig.update_geos(
        scope="north america", 
        center={"lat": 23.6345, "lon": -102.5528}, 
        projection_scale=4.5
    )
    return fig

def covid_comorbidity_heatmap():
    # Select comorbidities columns
    comorbidities = sorted(["DIABETES", "EPOC", "ASMA", "HIPERTENSION", "OBESIDAD", 
                     "RENAL_CRONICA", "TABAQUISMO", "OTRA_COM", "CARDIOVASCULAR",
                     'NEUMONIA', 'EMBARAZO', 'INMUSUPR', 'RENAL_CRONICA'])
    comorbidity_data = covid_df[comorbidities]
    
    # Correlation matrix heatmap
    fig = px.imshow(
        comorbidity_data.corr(), 
        text_auto=True, 
        title="Comorbidity Co-occurrence Heatmap", 
        color_continuous_scale="Blues",
        template="seaborn"
    )
    return fig

def covid_symptoms_to_admission():
    # Calculate interval from symptoms to admission
    covid_df['SYMPTOMS_TO_ADMISSION'] = (covid_df['FECHA_INGRESO'] - covid_df['FECHA_SINTOMAS']).dt.days
    
    # Histogram for the interval, cropped to 0-50 days
    fig = px.histogram(
        covid_df[covid_df['SYMPTOMS_TO_ADMISSION'] <= 50],  
        x='SYMPTOMS_TO_ADMISSION', 
        nbins=50,  
        title='Symptoms to Admission Interval (0-50 Days)',
        labels={'SYMPTOMS_TO_ADMISSION': 'Days', 'count': 'Number of Cases'},
        template='seaborn'
    )
    return fig

def dengue_age_gender_distribution():
    # Map gender to labels
    dengue_df['SEXO_LABEL'] = dengue_df['SEXO'].map({1: 'Female', 2: 'Male'})

    # Facet histogram for age distribution by gender
    fig = px.histogram(
        dengue_df,
        x='EDAD_ANOS',
        facet_col='SEXO_LABEL',
        color='SEXO_LABEL',
        title='Age Distribution by Gender',
        labels={'EDAD_ANOS': 'Age (Years)', 'SEXO_LABEL': 'Gender'},
        color_discrete_map={'Male': "#52BCEC", 'Female': "#006EC1"},
        nbins=30,
        template='seaborn'
    )
    return fig

def dengue_cases_over_time():
    # Extract month and year from symptoms date
    dengue_df['FECHA_SINTOMAS_MONTH'] = dengue_df['FECHA_SIGN_SINTOMAS'].dt.to_period('M').astype(str)

    # Group by month-year to calculate cases
    cases_by_time = dengue_df.groupby('FECHA_SINTOMAS_MONTH').size().reset_index(name='Cases')

    # Create combined bar and line chart
    fig = px.bar(
        cases_by_time,
        x='FECHA_SINTOMAS_MONTH',
        y='Cases',
        title='Dengue Cases Over Time',
        labels={'FECHA_SINTOMAS_MONTH': 'Date (Year-Month)', 'Cases': 'Number of Cases'},
        template='seaborn',
        color_discrete_sequence=['#89D1F3'],
    )
    fig.add_scatter(
        x=cases_by_time['FECHA_SINTOMAS_MONTH'],
        y=cases_by_time['Cases'],
        mode='lines',
        name='Trend',
        line=dict(color='#006EC1')
    )
    return fig

def dengue_hemorrhagic_cases_pie():
    hemorrhagic_counts = dengue_df['HEMORRAGICOS'].value_counts().reset_index()
    hemorrhagic_counts.columns = ['Hemorrhagic', 'Count']
    hemorrhagic_counts['Hemorrhagic'] = hemorrhagic_counts['Hemorrhagic'].map({2: 'Non-Hemorrhagic', 1: 'Hemorrhagic'})

    fig = px.pie(
        hemorrhagic_counts,
        names='Hemorrhagic',
        values='Count',
        title='Proportion of Hemorrhagic Dengue Cases',
        template='seaborn',
        color_discrete_sequence=['#006EC1', '#89D1F3']
    )
    return fig

def dengue_comorbidity_heatmap():
    # Select comorbidities columns
    comorbidities = ['DIABETES', 'HIPERTENSION', 'ENFERMEDAD_ULC_PEPTICA', 'INMUNOSUPR']
    comorbidity_data = dengue_df[comorbidities].copy()
    comorbidity_data = comorbidity_data.replace(2, 0)

    # Generate the heatmap
    fig = px.imshow(
        comorbidity_data.corr(), 
        text_auto=True,  
        title="Comorbidity Co-occurrence Heatmap", 
        color_continuous_scale="Blues", 
        labels={"color": "Correlation"},
        template="seaborn"
    )
    
    return fig

def febriles_parallel_categories():
    # Map variables to human-readable labels using catalogs
    febriles_df['SEXO_LABEL'] = febriles_df['SEXO'].map({1: 'Female', 2: 'Male'})
    febriles_df['VACUNACION_LABEL'] = febriles_df['VACUNACION'].map({1: 'Yes', 2: 'No'})
    febriles_df['DEFUNCION_LABEL'] = febriles_df['DEFUNCION'].map({1: 'Yes', 2: 'No'})
    febriles_df['COMPLICACIONES_LABEL'] = febriles_df['COMPLICACIONES'].map({1: 'Yes', 2: 'No'})
    
    fig = px.parallel_categories(
        febriles_df, 
        dimensions=['SEXO_LABEL', 'VACUNACION_LABEL', 'DEFUNCION_LABEL', 'COMPLICACIONES_LABEL'],
        title='Parallel Categories for Febriles Dataset',
        labels={
            'SEXO_LABEL': 'Gender',
            'VACUNACION_LABEL': 'Vaccination',
            'DEFUNCION_LABEL': 'Death',
            'COMPLICACIONES_LABEL': 'Complications'
        },
        template='seaborn',
        color='COMPLICACIONES',
        color_continuous_scale=px.colors.sequential.Blues
        
    )
    fig.update_layout(coloraxis_showscale=False)

    return fig

def febriles_strip_plot():
    # Map categorical variables to labels
    febriles_df['VACUNACION_LABEL'] = febriles_df['VACUNACION'].map({2: 'No', 1: 'Yes'})
    febriles_df['COMPLICACIONES_LABEL'] = febriles_df['COMPLICACIONES'].map({2: 'No', 1: 'Yes'})

    # Create a strip plot
    fig = px.strip(
        febriles_df,
        x='VACUNACION_LABEL',
        y='EDAD_ANOS',
        color='COMPLICACIONES_LABEL',
        title='Age vs. Vaccination and Complications',
        labels={
            'VACUNACION_LABEL': 'Vaccination Status',
            'EDAD_ANOS': 'Age (Years)',
            'COMPLICACIONES_LABEL': 'Complications'
        },
        template='seaborn',
        stripmode='overlay',
        color_discrete_map={'Yes': '#B5E5F9', 'No': '#006EC1'}
    )
    fig.update_traces(jitter=0.3, marker=dict(size=8))

    return fig

def febriles_geospatial_distribution():
    # Map state codes to names
    febriles_df['STATE_LABEL'] = febriles_df['ENTIDAD_RES'].map(entidades_mapping)
    
    # Group by state for case counts
    cases_by_state = febriles_df.groupby('STATE_LABEL').size().reset_index(name='Case_Count')
    
    # Create choropleth map
    fig = px.choropleth(
        cases_by_state,
        geojson=mexico,  # Use GeoJSON for Mexican states
        locations='STATE_LABEL',
        featureidkey='properties.name',
        color='Case_Count',
        title='Geospatial Distribution of Febriles Cases',
        color_continuous_scale='Blues',
        labels={'Case_Count': 'Cases'},
        range_color=(0, 1500),
        template='seaborn'
    )
    fig.update_geos(
        scope='north america',
        center={'lat': 23.6345, 'lon': -102.5528},
        projection_scale=4.5,
    )
    return fig

morbilidad_df[['state', 'sickness']] = morbilidad_df[['state', 'sickness']].apply(lambda x: x.str.title())

def morbilidad_top_diseases_over_time():
    # Group data by year and sickness, summing up the cases
    top_diseases = (
        morbilidad_df.groupby(['year', 'sickness'])['value'].sum()
        .reset_index()
        .sort_values(by='value', ascending=False)
    )

    # Select the top 10 diseases overall
    top_10_diseases = top_diseases.groupby('sickness')['value'].sum().nlargest(10).index

    # Filter data to include only top 10 diseases
    filtered_data = top_diseases[top_diseases['sickness'].isin(top_10_diseases)]

    # Create a grouped bar chart
    fig = px.bar(
        filtered_data,
        x='year',
        y='value',
        color='sickness',
        barmode='group', 
        title='Top 10 Diseases Over Time',
        labels={'year': 'Year', 'value': 'Cases', 'sickness': 'Disease'},
        template='seaborn',
        color_discrete_sequence=px.colors.cyclical.Phase
    )

    # Update layout for better appearance
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Cases",
        legend_title="Disease",
    )

    return fig

# Define mapping for duplicate or inconsistent state names
state_mapping = {
    'Distrito Federal': 'Ciudad de México',
    'Sanluis Potosi': 'San Luis Potosí',
    'Queretaro': 'Querétaro',
    'Mexico': 'México',
    'Michoacan': 'Michoacán',
    'Nuevo Leon': 'Nuevo León',
    'Yucatan': 'Yucatán',
    'Coahuila': 'Coahuila',
    'Veracruz': 'Veracruz',
    'Ciudad De Mexico': 'Ciudad de México',
    'San Luis Potosi': 'San Luis Potosí',
    'Total Global': None  # Remove non-geographic entry
}

# Apply mapping to standardize state names
morbilidad_df['state'] = morbilidad_df['state'].replace(state_mapping)
invalid_states = {'Estados Unidos Mexicanos', 'Se Ignora', 'No Especificado', 'No Aplica', None}
morbilidad_df = morbilidad_df[~morbilidad_df['state'].isin(invalid_states)]

def morbilidad_treemap():
    # Aggregate cases by state and sickness
    treemap_data = (
        morbilidad_df.groupby(['state', 'sickness'])['value'].sum()
        .reset_index()
        .sort_values(by='value', ascending=False)
    )

    treemap_data = treemap_data[treemap_data['value'] > 0]
    
    # Create a treemap
    fig = px.treemap(
        treemap_data,
        path=['state', 'sickness'],
        values='value',
        title='Distribution of Cases by State and Disease',
        labels={'state': 'State', 'sickness': 'Disease', 'value': 'Cases'},
        template='seaborn',
        color='value',
        range_color=(0, 80000000),
        color_continuous_scale=px.colors.sequential.Blues,
    )

    return fig

def morbilidad_geospatial_distribution():
    # Aggregate cases by state
    cases_by_state = (
        morbilidad_df.groupby('state')['value'].sum()
        .reset_index()
        .rename(columns={'value': 'Total_Cases'})
    )

    # Create a choropleth map
    fig = px.choropleth(
        cases_by_state,
        geojson=mexico,  
        locations='state',
        featureidkey='properties.name',
        color='Total_Cases',
        title='Geospatial Distribution of Total Cases',
        labels={'state': 'State', 'Total_Cases': 'Total Cases'},
        color_continuous_scale='Blues',
        template='seaborn',
    )
    fig.update_geos(
        scope='north america',
        center={'lat': 23.6345, 'lon': -102.5528},
        projection_scale=4.5,
    )

    return fig


"""_Statistics_
This section contains the statistical analysis functions for the dashboard.
"""

def anova_age_across_datasets():
    """
    Perform ANOVA on age across datasets (COVID, Dengue, Febriles) and visualize the results.
    """
    # Prepare data
    covid_age = covid_df['EDAD']
    dengue_age = dengue_df['EDAD_ANOS']
    febriles_age = febriles_df['EDAD_ANOS']

    # Perform ANOVA
    anova_result = f_oneway(covid_age, dengue_age, febriles_age)
    
    # Create a summary text
    summary_text = (
        f"ANOVA Results:\n"
        f"F-statistic: {anova_result.statistic:.2f}\n"
        f"P-value: {anova_result.pvalue:.4f}"
    )

    # Boxplot for visualizing age distribution across datasets
    all_ages = pd.DataFrame({
        'Dataset': ['COVID'] * len(covid_age) + ['Dengue'] * len(dengue_age) + ['Febriles'] * len(febriles_age),
        'Age': pd.concat([covid_age, dengue_age, febriles_age])
    })

    fig = px.box(
        all_ages,
        x='Dataset',
        y='Age',
        title="Age Distribution Across Datasets (ANOVA)",
        labels={'Age': 'Age (Years)', 'Dataset': 'Dataset'},
        template='seaborn',
        color='Dataset',
        color_discrete_map={'COVID': '#006EC1', 'Dengue': '#52BCEC', 'Febriles': '#89D1F3'},
    )
    
    # Add ANOVA summary text
    fig.add_annotation(
        x=1.5, y=max(all_ages['Age']),
        text=summary_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left",
        bgcolor="white",
        bordercolor="black",
    )

    return fig

def manova_covid_dengue():
    """
    Perform MANOVA on COVID and Dengue datasets for comorbidities and return a summary table.
    """
    # Combine datasets with selected variables
    selected_columns = ['DIABETES', 'HIPERTENSION', 'EMBARAZO']
    covid_comorbid = covid_df[selected_columns].assign(Dataset='COVID')
    dengue_comorbid = dengue_df[selected_columns].assign(Dataset='Dengue')

    combined_data = pd.concat([covid_comorbid, dengue_comorbid])

    # Perform MANOVA
    manova = MANOVA.from_formula('DIABETES + HIPERTENSION + EMBARAZO ~ Dataset', data=combined_data)
    manova_result = manova.mv_test()

    # Extract MANOVA results into a readable DataFrame
    dataset_results = manova_result.results['Dataset']['stat']
    summary_table = dataset_results.reset_index().rename(columns={
        'index': 'Test',
        'Value': 'Statistic',
        'F Value': 'F-value',
        'Num DF': 'Num DF',
        'Den DF': 'Den DF',
        'Pr > F': 'p-value'
    })

    # Filter to keep only the first two rows
    summary_table = summary_table.iloc[:2]
    
    # Round results
    numeric_cols = ['Statistic', 'F-value', 'Num DF', 'Den DF', 'p-value']
    summary_table[numeric_cols] = summary_table[numeric_cols].apply(pd.to_numeric).round(3)
    
    return summary_table
    
def manova_covid_single():
    """
    Perform MANOVA on the COVID dataset for comorbidities by TIPO_PACIENTE (Ambulatory vs Hospitalized).
    """
    # Filter COVID data for valid TIPO_PACIENTE values
    covid_filtered = covid_df[covid_df['TIPO_PACIENTE'].isin([1, 2])]

    # Perform MANOVA
    manova = MANOVA.from_formula('DIABETES + HIPERTENSION + ASMA + EPOC + OBESIDAD ~ TIPO_PACIENTE', data=covid_filtered)
    manova_result = manova.mv_test()

    # Extract MANOVA results into a readable DataFrame
    tipo_paciente_results = manova_result.results['TIPO_PACIENTE']['stat']
    summary_table = tipo_paciente_results.reset_index().rename(columns={
        'index': 'Test',
        'Value': 'Statistic',
        'F Value': 'F-value',
        'Num DF': 'Num DF',
        'Den DF': 'Den DF',
        'Pr > F': 'p-value'
    })
    
    # Filter to keep only the first two rows
    summary_table = summary_table.iloc[:2]

    # Round results
    numeric_cols = ['Statistic', 'F-value', 'Num DF', 'Den DF', 'p-value']
    summary_table[numeric_cols] = summary_table[numeric_cols].apply(pd.to_numeric).round(3)

    return summary_table



"""_Cluster_
This section contains the clustering functions for the dashboard.
"""

def kmeans_dengue_clustering():
    """
    Perform K-Means clustering on numerical variables in the Dengue dataset with improved visualization.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Select relevant numerical columns and preprocess
    features = ['EDAD_ANOS', 'FECHA_SIGN_SINTOMAS', 'DIABETES', 'HIPERTENSION']
    dengue_features = dengue_df[features].copy()
    dengue_features['FECHA_SIGN_SINTOMAS'] = (
        (dengue_df['FECHA_SIGN_SINTOMAS'] - dengue_df['FECHA_SIGN_SINTOMAS'].min()).dt.days
    )
    dengue_features = dengue_features.fillna(0)  # Handle NaNs

    # Standardize data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dengue_features)

    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    dengue_df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Map clusters to distinct colors
    cluster_colors = {0: '0', 1: '1', 2: '2'}
    dengue_df['Cluster_Color'] = dengue_df['Cluster'].map(cluster_colors)

    # Create scatter plot with discrete colors
    fig = px.scatter(
        dengue_df,
        x='EDAD_ANOS',
        y='FECHA_SIGN_SINTOMAS',
        color='Cluster_Color',
        color_discrete_map={'0': '#006EC1', '1': '#009EE5', '2': '#89D1F3'},
        title="K-Means Clustering of Dengue Cases",
        labels={'EDAD_ANOS': 'Age (Years)', 'FECHA_SIGN_SINTOMAS': 'Days Since First Symptom', 'Cluster_Color': 'Cluster'},
        template="seaborn"
    )

    # Add cluster labels
    for cluster in dengue_df['Cluster'].unique():
        cluster_data = dengue_df[dengue_df['Cluster'] == cluster]
        mean_x = cluster_data['EDAD_ANOS'].mean()
        mean_y = cluster_data['FECHA_SIGN_SINTOMAS'].mean()
        fig.add_annotation(
            x=mean_x,
            y=mean_y,
            text=str(cluster),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black',
            font=dict(color='black', size=12),
            bgcolor="white",
            bordercolor="black"
        )
        
    # Add annotation explaining variable roles
    fig.add_annotation(
        text=(
            "Indicators used for clustering:<br>"
            "- Diabetes <br>" 
            "- Hypertension"
        ),
        xref="paper", yref="paper",
        x=0.95, y=1.2,  # Position the annotation outside the plot
        showarrow=False,
        align="left",
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black"
    )

    return fig

def hierarchical_clustering_covid():
    """
    Perform hierarchical clustering on the COVID dataset using an optimized linkage method,
    formatted to align with the dashboard's aesthetics.
    """
    # Select features for clustering
    features = ['TIPO_PACIENTE', 'DIABETES', 'HIPERTENSION', 'ASMA', 'EPOC', 'OBESIDAD']
    covid_features = covid_df[features].fillna(0)

    # Downsample the data
    sample_size = 1000
    covid_features_sampled = covid_features.sample(n=sample_size, random_state=42)

    # Standardize features
    scaled_features = StandardScaler().fit_transform(covid_features_sampled)

    # Perform hierarchical clustering
    linkage_matrix = linkage(scaled_features, method='ward')

    # Create dendrogram with updated aesthetics
    plt.figure(figsize=(10, 7))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=20,
        show_leaf_counts=True,
        leaf_font_size=10,  # Match font size to dashboard's readability
        above_threshold_color="#006EC1",  # Consistent blue for unclustered segments
        color_threshold=30,  # Adjust clustering cut-off color
    )
    plt.title("Hierarchical Clustering (Ward Linkage) for COVID Dataset", fontsize=16, color="#333333", pad=15)
    plt.xlabel("Cluster Size", fontsize=14, color="#333333", labelpad=10)
    plt.ylabel("Distance", fontsize=14, color="#333333", labelpad=10)
    plt.xticks(fontsize=12, color="#333333")  # Match axis tick labels with Plotly styling
    plt.yticks(fontsize=12, color="#333333")
    plt.grid(axis='y', linestyle='--', alpha=0.6)  # Add light grid for readability
    plt.tight_layout()

    # Save the figure to the assets directory
    plt.savefig('assets/hierarchical_clustering_covid_fast.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # Return dendrogram as an image component for Dash
    fig = html.Div([
        html.Img(
            src='/assets/hierarchical_clustering_covid_fast.png',
            style={
                'height': 'auto',
                'width': '100%',
                'object-fit': 'contain',
                'border': '1px solid #ccc',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            }
        ),
        html.P(
            "Features used for clustering: TIPO_PACIENTE, Diabetes, Hypertension, Asthma, EPOC, Obesity",
            style={'text-align': 'center', 'margin-top': '10px', 'font-size': '14px'}
        )
    ])
    return fig

def kprototypes_febriles():
    """
    Perform K-Prototypes clustering on Febriles dataset with categorical and numerical data.
    """
    # Prepare data
    features = ['VACUNACION_LABEL', 'DEFUNCION_LABEL', 'COMPLICACIONES_LABEL', 'EDAD_ANOS']
    febriles_features = febriles_df[features].copy()

    # Encode categorical variables as integers
    categorical_columns = ['VACUNACION_LABEL', 'DEFUNCION_LABEL', 'COMPLICACIONES_LABEL']
    febriles_features[categorical_columns] = febriles_features[categorical_columns].apply(lambda col: col.astype('category').cat.codes)

    # Downsample the dataset
    sample_size = 1000 
    febriles_sampled = febriles_features.sample(n=sample_size, random_state=42)
    
    # Convert to numpy array for K-Prototypes
    data = febriles_sampled.values

    # Apply K-Prototypes
    kproto = KPrototypes(n_clusters=3, init='Cao', gamma=0.5, random_state=42)
    clusters = kproto.fit_predict(data, categorical=[0, 1, 2])

    # Add cluster labels to the dataframe
    febriles_sampled['Cluster'] = clusters

    # Create scatter plot for clustering
    fig = px.scatter(
        febriles_sampled,
        x='EDAD_ANOS',
        y='Cluster',
        color='Cluster',
        title="K-Prototypes Clustering on Febriles Dataset",
        labels={'EDAD_ANOS': 'Age (Years)', 'Cluster': 'Cluster'},
        template="seaborn",
        color_discrete_map={0: '#006EC1', 1: '#009EE5', 2: '#89D1F3'}
    )
    
    # Add annotation explaining variable roles
    fig.add_annotation(
        text=(
            "Indicators used for clustering:<br>"
            "- Vaccination <br>" 
            "- Death <br>"
            "- Complications"
        ),
        xref="paper", yref="paper",
        x=0.95, y=1.25,  
        showarrow=False,
        align="left",
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black"
    )
    return fig

def time_series_clustering_morbidities():
    """
    Perform time-series clustering on the morbidities dataset and create a Plotly line plot showing only cluster mean trends.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Aggregate and Pivot Data
    aggregated_data = (
        morbilidad_df.groupby(['sickness', 'year'])['value']
        .sum()
        .reset_index()
    )
    pivoted_data = aggregated_data.pivot(index='sickness', columns='year', values='value').fillna(0)

    # Scale Data
    scaler = TimeSeriesScalerMeanVariance()
    scaled_time_series = scaler.fit_transform(pivoted_data.values)

    # Apply Time-Series K-Means Clustering
    n_clusters = 3
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    clusters = model.fit_predict(scaled_time_series)

    # Add Clusters to the Data and Aggregate Mean Trends
    pivoted_data['Cluster'] = clusters
    mean_trends = pivoted_data.groupby('Cluster').mean().reset_index().melt(
        id_vars=['Cluster'], var_name='Year', value_name='Mean Value'
    )

    # Create Line Plot
    fig = px.line(
        mean_trends,
        x='Year',
        y='Mean Value',
        color='Cluster',
        title='Mean Time-Series Trends for Morbidities by Cluster',
        labels={'Mean Value': 'Average Cases', 'Year': 'Year', 'Cluster': 'Cluster'},
        template='seaborn',
        color_discrete_sequence={0: '#006EC1', 1: '#009EE5', 2: '#89D1F3'}
    )

    return fig

def pca_covid():
    """
    Perform PCA on the COVID dataset for dimensionality reduction and visualize the top 2 components.
    """
    # Select features for PCA
    features = ['EDAD', 'DIABETES', 'HIPERTENSION', 'ASMA', 'EPOC', 'OBESIDAD']
    covid_features = covid_df[features].fillna(0)

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(covid_features)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_features)

    # Add PCA components back to the dataframe
    covid_df['PCA1'] = pca_results[:, 0]
    covid_df['PCA2'] = pca_results[:, 1]

    patient_colors = {1: 'Ambulatory', 2: 'Hospitalized'}
    covid_df['TIPO_PACIENTE_Label'] = covid_df['TIPO_PACIENTE'].map(patient_colors)

    # Create scatter plot for PCA results
    fig = px.scatter(
        covid_df,
        x='PCA1',
        y='PCA2',
        title="PCA on COVID Dataset",
        color='TIPO_PACIENTE_Label',  # Use patient type for coloring
        labels={
            'PCA1': 'Principal Component 1',
            'PCA2': 'Principal Component 2',
            'TIPO_PACIENTE_Label': 'Patient Type'
        },
        template="seaborn",
        color_discrete_map={
            'Ambulatory': '#006EC1',
            'Hospitalized': '#89D1F3'
        },
    )

    # Add explained variance as annotations
    explained_variance = pca.explained_variance_ratio_ * 100
    fig.add_annotation(
        text=f"Explained Variance:<br>PC1: {explained_variance[0]:.2f}%<br>PC2: {explained_variance[1]:.2f}%",
        xref="paper", yref="paper",
        x=1.16, y=0.5,
        showarrow=False,
        align="left",
        font=dict(size=12)
    )

    return fig

def pca_morbidities_3d():
    """
    Perform PCA on the morbidities dataset for dimensionality reduction and visualize the top 3 components in 3D.
    """
    # Pivot the morbidities dataset to create a matrix with states as rows and diseases as columns
    morbidities_pivot = morbilidad_df.pivot_table(
        index='state', columns='sickness', values='value', aggfunc='sum', fill_value=0
    )

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(morbidities_pivot)

    # Apply PCA
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(scaled_data)

    # Add PCA components back to the DataFrame
    pca_df = pd.DataFrame(
        pca_results,
        index=morbidities_pivot.index,
        columns=['PCA1', 'PCA2', 'PCA3']
    )

    # Create a numeric column for coloring
    pca_df['State_Index'] = pd.factorize(pca_df.index)[0]
    
    # Explained variance for annotations
    explained_variance = pca.explained_variance_ratio_ * 100

    # Create a 3D scatter plot for PCA results
    fig = px.scatter_3d(
        pca_df,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        title="PCA on Morbidities Dataset by State",
        color='State_Index', 
        text=pca_df.index,
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2', 'PCA3': 'Principal Component 3'},
        template="seaborn",
        color_continuous_scale=px.colors.sequential.Blues
    )

    # Change font size for annotations
    fig.update_traces(textfont=dict(size=8))
    
    # Add explained variance annotation in the title
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text=f"PC1 ({explained_variance[0]:.2f}%)",
                    font=dict(size=10, style='italic')
                ),
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title=dict(
                    text=f"PC2 ({explained_variance[1]:.2f}%)",
                    font=dict(size=10, style='italic')
                ),
                tickfont=dict(size=10)
            ),
            zaxis=dict(
                title=dict(
                    text=f"PC3 ({explained_variance[2]:.2f}%)",
                    font=dict(size=10, style='italic')
                ),
                tickfont=dict(size=10)
            ),
        ),
        coloraxis_showscale=False#,
       #margin=dict(l=20, r=20, t=40, b=20)  # Adjust the margins as needed
    )
     
    return fig

def factor_analysis_covid():
    """
    Perform Factor Analysis on the COVID dataset using the factor_analyzer library with Varimax rotation.
    """
    # Select features for Factor Analysis
    features = ['DIABETES', 'HIPERTENSION', 'OBESIDAD', 'ASMA', 'EPOC', 
                'INMUSUPR', 'CARDIOVASCULAR', 'RENAL_CRONICA', 'TABAQUISMO']
    covid_features = covid_df[features].fillna(0)

    # Standardize the data
    standardized_data = zscore(covid_features)

    # Apply Factor Analysis
    n_factors = 2  # Number of factors to extract
    fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method='principal')
    fa.fit(standardized_data)

    # Extract unrotated factor loadings
    loadings = fa.loadings_

    # Apply Varimax Rotation
    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(loadings)

    # Create a DataFrame for rotated loadings
    loading_df = pd.DataFrame(
        rotated_loadings,
        index=features,  # Features as rows
        columns=[f"Factor {i+1}" for i in range(n_factors)]  # Factors as columns
    )

    # Create heatmap for rotated loadings
    fig = ff.create_annotated_heatmap(
        z=loading_df.values,
        x=loading_df.columns.tolist(),  # Factors
        y=loading_df.index.tolist(),   # Features
        annotation_text=np.round(loading_df.values, 2),  # Rounded values for annotations
        colorscale='Blues',
        showscale=False
    )
    fig.update_layout(
        title="Rotated Factor Loadings (Varimax) for COVID Dataset",
        xaxis_title="",
        yaxis_title="Features",
        template="seaborn"
    )

    return fig

def factor_analysis_dengue():
    """
    Perform Factor Analysis on the Dengue dataset using the factor_analyzer library with Varimax rotation.
    """
    # Select features for Factor Analysis
    features = ['DIABETES', 'HIPERTENSION', 'ENFERMEDAD_ULC_PEPTICA', 
                'ENFERMEDAD_RENAL', 'INMUNOSUPR', 'CIRROSIS_HEPATICA']
    dengue_features = dengue_df[features].fillna(0)

    # Standardize the data
    standardized_data = zscore(dengue_features)

    # Apply Factor Analysis
    n_factors = 2  # Number of factors to extract
    fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method='principal')
    fa.fit(standardized_data)

    # Extract unrotated factor loadings
    loadings = fa.loadings_

    # Apply Varimax Rotation
    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(loadings)

    # Create a DataFrame for rotated loadings
    loading_df = pd.DataFrame(
        rotated_loadings,
        index=features,  # Features as rows
        columns=[f"Factor {i+1}" for i in range(n_factors)]  # Factors as columns
    )

    # Create heatmap for rotated loadings
    fig = ff.create_annotated_heatmap(
        z=loading_df.values,
        x=loading_df.columns.tolist(),  # Factors
        y=loading_df.index.tolist(),   # Features
        annotation_text=np.round(loading_df.values, 2),  # Rounded values for annotations
        colorscale='Blues',
        showscale=False
    )
    fig.update_layout(
        title="Rotated Factor Loadings (Varimax) for Dengue Dataset",
        xaxis_title="",
        yaxis_title="Features",
        template="seaborn"
    )

    return fig

def factor_analysis_febriles():
    """
    Perform Factor Analysis on the Febriles dataset using the factor_analyzer library with Varimax rotation.
    """
    # Select features for Factor Analysis
    features = ['COMPLICACIONES', 'DEFUNCION', 'VACUNACION', 'DIAGNOSTICO']
    febriles_features = febriles_df[features].fillna(0)

    # Standardize the data
    standardized_data = zscore(febriles_features)

    # Apply Factor Analysis
    n_factors = 2  # Number of factors to extract
    fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method='principal')
    fa.fit(standardized_data)

    # Extract unrotated factor loadings
    loadings = fa.loadings_

    # Apply Varimax Rotation
    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(loadings)

    # Create a DataFrame for rotated loadings
    loading_df = pd.DataFrame(
        rotated_loadings,
        index=features,  # Features as rows
        columns=[f"Factor {i+1}" for i in range(n_factors)]  # Factors as columns
    )

    # Define a custom colormap from the middle to the upper end of `Blues`
    blues_colormap = [[0.0, 'rgb(190, 220, 240)'],  # Lighter blue
                      [0.5, 'rgb(100, 150, 200)'],  # Mid blue
                      [1.0, 'rgb(0, 70, 150)']]    # Darker blue
    
    # Create heatmap for rotated loadings
    fig = ff.create_annotated_heatmap(
        z=loading_df.values,
        x=loading_df.columns.tolist(),  # Factors
        y=loading_df.index.tolist(),   # Features
        annotation_text=np.round(loading_df.values, 2),  # Rounded values for annotations
        colorscale=blues_colormap,
        showscale=True
    )
    fig.update_layout(
        title="Rotated Factor Loadings (Varimax) for Febriles Dataset",
        xaxis_title="",
        yaxis_title="Features",
        template="seaborn"
    )

    return fig



"""_Classification_
This section contains the classification functions for the dashboard.
"""

def logistic_regression_tipo_paciente():
    """
    Logistic Regression for predicting TIPO_PACIENTE (Ambulatorio vs Hospitalized)
    using comorbidities in the COVID dataset.
    """
    # Filter the dataset for necessary columns and valid TIPO_PACIENTE values
    covid_filtered = covid_df[covid_df['TIPO_PACIENTE'].isin([1, 2])].copy()
    features = ['DIABETES', 'HIPERTENSION', 'OBESIDAD']
    target = 'TIPO_PACIENTE'

    # Handle missing values (assume NA = 2 for "No" in this case)
    covid_filtered[features] = covid_filtered[features].fillna(2)

    # Split the data into training and testing sets
    X = covid_filtered[features]
    y = covid_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Logistic Regression model
    model = LogisticRegression(random_state=69, max_iter=200)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Ambulatorio', 'Hospitalizado'])
    cm = confusion_matrix(y_test, y_pred)

    # Feature Importance (Coefficients)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': np.round(model.coef_[0], 3)
    }).sort_values(by='Coefficient', ascending=False)
    
    return model, acc, report, cm, feature_importance

def random_forest_clasificacion_final():
    """
    Random Forest for predicting CLASIFICACION_FINAL (e.g., positive, negative, suspected)
    using comorbidities in the COVID dataset.
    """
    # Filter the dataset for necessary columns
    covid_filtered = covid_df[covid_df['CLASIFICACION_FINAL'].isin(range(1, 8))].copy()
    features = ['DIABETES', 'HIPERTENSION', 'OBESIDAD', 'ASMA', 'EPOC', 'INMUSUPR']
    target = 'CLASIFICACION_FINAL'

    # Handle missing values (assume NA = 2 for "No" in this case)
    covid_filtered[features] = covid_filtered[features].fillna(2)

    # Split the data into training and testing sets
    X = covid_filtered[features]
    y = covid_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)

    # Initialize Random Forest model
    model = RandomForestClassifier(random_state=42, class_weight=None, max_depth=5, min_samples_leaf=1, min_samples_split=2, n_estimators=50)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
        'Confirmed by Association', 'Confirmed by Decease', 'Confirmed by Laboratory',
        'Invalid', 'Not Applicable', 'Suspected', 'Negative'
    ], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return model, acc, report, cm, feature_importance, X_test, y_test, y_pred_proba

def precision_recall_curve_rf(y_test, y_pred_proba, n_classes=7):
    """
    Generate a Precision-Recall curve for Random Forest predictions.
    Handles cases where no positive samples exist for a class.
    """
    precision = {}
    recall = {}
    fig = go.Figure()

    for i in range(n_classes):
        # Check if the class has any positive samples in y_test
        if (y_test == i).sum() == 0:
            continue  # Skip classes with no positive samples

        # Compute precision-recall curve
        precision[i], recall[i], _ = precision_recall_curve(
            (y_test == i).astype(int), y_pred_proba[:, i]
        )

        # Add curve to the figure
        fig.add_trace(go.Scatter(
            x=recall[i], y=precision[i],
            mode='lines',
            name=f'Class {i}'
        ))

    # Update layout
    fig.update_layout(
        title='Precision-Recall Curve (Random Forest)',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='seaborn'
    )

    return fig

def extract_weighted_metrics(report, accuracy):
    """
    Extract accuracy, weighted precision, recall, and F1-score from a classification report.
    """
    report_lines = report.split("\n")
    weighted_line = report_lines[-2].split()  # Weighted Avg is usually the second-to-last line

    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [accuracy, float(weighted_line[-4]), float(weighted_line[-3]), float(weighted_line[-2])]
    }
    return pd.DataFrame(metrics)


"""
def gradient_boosting_resultado_pcr():

    #Gradient Boosting for predicting RESULTADO_PCR (positive/negative/other)
    #using demographic and temporal features in the Dengue dataset.

    # Filter Dengue dataset for necessary columns
    dengue_filtered = dengue_df[dengue_df['RESULTADO_PCR'].isin([1, 2, 3, 4, 5])].copy()
    features = ['EDAD_ANOS', 'DIABETES', 'HIPERTENSION', 'HEMORRAGICOS', 'FECHA_SIGN_SINTOMAS']
    target = 'RESULTADO_PCR'

    # Preprocess data
    dengue_filtered['FECHA_SIGN_SINTOMAS'] = (
        (dengue_filtered['FECHA_SIGN_SINTOMAS'] - dengue_filtered['FECHA_SIGN_SINTOMAS'].min()).dt.days
    )  # Convert dates to numerical days
    dengue_filtered[features] = dengue_filtered[features].fillna(0)  # Fill missing values

    # Split data into training and testing sets
    X = dengue_filtered[features]
    y = dengue_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize Gradient Boosting model
    model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
        'Positive', 'Negative', 'Invalid', 'Suspected', 'Other'
    ])
    cm = confusion_matrix(y_test, y_pred)

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return model, acc, report, cm, feature_importance

def save_gradient_boosting_results():
    # Generate results
    model, acc, report, cm, feature_importance = gradient_boosting_resultado_pcr()

    # Prepare data to save
    results = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance.to_dict(orient="records")
    }

    # Save to JSON
    with open("assets/gradient_boosting_results.json", "w") as f:
        json.dump(results, f, indent=4)

save_gradient_boosting_results()
"""

def load_gradient_boosting_results():
    # Load JSON file
    with open("assets/gradient_boosting_results.json", "r") as f:
        results = json.load(f)
    
    # Parse results
    acc = results["accuracy"]
    report = results["classification_report"]
    cm = results["confusion_matrix"]
    feature_importance = pd.DataFrame(results["feature_importance"])

    return acc, report, cm, feature_importance

def random_forest_dictamen():
    """
    Random Forest for predicting DICTAMEN using demographic and medical features in the Dengue dataset.
    """
    # Filter Dengue dataset for necessary columns
    dengue_filtered = dengue_df[dengue_df['DICTAMEN'].isin([1, 2, 3, 4])].copy()
    features = ['EDAD_ANOS', 'ENTIDAD_RES', 'DIABETES', 'HIPERTENSION', 'HEMORRAGICOS']
    target = 'DICTAMEN'

    # Preprocess data
    dengue_filtered[features] = dengue_filtered[features].fillna(0)  # Fill missing values

    # Split data into training and testing sets
    X = dengue_filtered[features]
    y = dengue_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)

    # Initialize Random Forest model
    model = RandomForestClassifier(random_state=42, max_depth=15, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Required for ROC curve

    # Dynamically set target names and labels based on the unique classes in y
    unique_classes = sorted(y.unique())
    target_names = [f"Class {cls}" for cls in unique_classes]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)  # Use `labels` to match classes exactly

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return model, acc, report, cm, feature_importance, X_test, y_test, y_pred_proba

def roc_curve_rf(y_test, y_pred_proba, n_classes=None):
    """
    Generate an ROC Curve for Random Forest predictions.
    Dynamically handles cases where no positive samples exist for a class.
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go

    # Dynamically determine the number of classes from y_test and y_pred_proba
    if n_classes is None:
        n_classes = y_pred_proba.shape[1]

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    fig = go.Figure()

    for i in range(n_classes):
        # Check if the class exists in y_test
        if (y_test == i).sum() == 0:
            continue  # Skip classes with no positive samples

        # Compute ROC curve
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Add curve to the figure
        fig.add_trace(go.Scatter(
            x=fpr[i], y=tpr[i],
            mode='lines',
            line=dict(color=['#006EC1', '#006EC1'][i % 2]), 
            name=f'Class {i} (AUC = {roc_auc[i]:.2f})'
        ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='#89D1F3'),
        name='No Skill'
    ))

    # Update layout
    fig.update_layout(
        title='ROC Curve (Random Forest)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='seaborn'
    )

    return fig

def qda_classification_dengue():
    """
    Perform QDA classification for DEFUNCION on the Dengue dataset and evaluate the model.
    """
    # Select features and target variable
    features = ['EDAD_ANOS', 'DIABETES', 'HIPERTENSION', 'INMUNOSUPR', 'EMBARAZO']
    target = 'DEFUNCION'
    
    # Prepare the data
    X = dengue_df[features].fillna(0)
    y = dengue_df[target]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize QDA model
    qda_model = QuadraticDiscriminantAnalysis()

    # Train the model
    qda_model.fit(X_train, y_train)

    # Make predictions
    y_pred = qda_model.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Alive', 'Deceased'])
    cm = confusion_matrix(y_test, y_pred)

    # Compute pseudo-feature importances for QDA (based on absolute mean differences in the means of the features per class)
    class_means = qda_model.means_
    importance = np.abs(class_means[0] - class_means[1])  # Differences between class means
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    return qda_model, acc, report, cm, feature_importance

def qda_dengue_defuncion():
    """
    Apply Quadratic Discriminant Analysis (QDA) to the Dengue dataset for DEFUNCION prediction.
    """
    # Select features and target variable
    features = ['EDAD_ANOS', 'DIABETES', 'HIPERTENSION', 'CIRROSIS_HEPATICA', 'INMUNOSUPR']
    dengue_filtered = dengue_df.dropna(subset=features + ['DEFUNCION'])
    X = dengue_filtered[features]
    y = dengue_filtered['DEFUNCION']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_scaled, y)
    
    # Compute QDA scores
    qda_scores = qda.decision_function(X_scaled)
    dengue_filtered['QDA1'] = qda_scores if qda_scores.ndim == 1 else qda_scores[:, 0]
    
    qda_colors = {1: 'Deceased', 2: 'Alive'}
    dengue_filtered['DEFUNCION_Label'] = dengue_filtered['DEFUNCION'].map(qda_colors)

    # Visualization
    fig = px.scatter(
        dengue_filtered,
        x='QDA1',
        y=dengue_filtered.index,
        color='DEFUNCION_Label',
        title="QDA Projection for DEFUNCION in Dengue Dataset",
        labels={'color': 'DEFUNCION', 'QDA1': 'QDA Component 1', 'DEFUNCION_Label': 'DEFUNCION'},
        template='seaborn',
        color_discrete_map={'Deceased': '#006EC1', 'Alive': '#52BCEC'}
    )

    # Update layout
    fig.update_layout(
        #coloraxis_showscale=False,  # Hide color bar
        xaxis_title="QDA1",
        yaxis_title="Index"
    )

    return fig



"""_Dash_
This section contains the functions for the Dash web application.
"""

eda_tab = dbc.Tab(
    label="Exploratory Data Analysis",
    children=[
        # COVID-19 Subtitle and Visualizations
        html.H4("COVID-19 Dataset", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section presents exploratory data analysis (EDA) for the COVID-19 dataset, "
            "covering demographic, temporal, geospatial, and medical insights.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),  

        # Row for Age and Gender Distribution, and Cases Over Time
        dbc.Row([
            dbc.Col(dcc.Graph(id='age-gender-distribution', figure=covid_age_gender_distribution(), 
                              style={'height': '400px'}), width=6),
            dbc.Col(dcc.Graph(id='cases-over-time', figure=covid_cases_over_time(), 
                              style={'height': '400px'}), width=6),
        ], className="mb-4", style={'margin-bottom': '15px'}), 

        # Row for Geospatial Distribution of Cases
        dbc.Row([
            dbc.Col(dcc.Graph(id='geospatial-cases', figure=covid_cases_geospatial(), 
                              style={'height': '800px', 'width': '100%'}), width=10),
        ], className="justify-content-center mb-4", style={'margin-bottom': '30px'}),

        # Row for Comorbidity Heatmap and Symptoms-to-Admission Interval
        dbc.Row([
            dbc.Col(dcc.Graph(id='comorbidity-heatmap', figure=covid_comorbidity_heatmap(), 
                              style={'height': '700px'}), width=6),
            dbc.Col(dcc.Graph(id='symptoms-to-admission', figure=covid_symptoms_to_admission(), 
                              style={'height': '500px'}), width=6),
        ], className="mb-4", style={'margin-bottom': '30px'}),
        
        # Dengue Subtitle
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        html.H4("Dengue Dataset", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section presents exploratory data analysis (EDA) for the Dengue dataset, "
            "covering demographic, temporal, geospatial, and medical insights.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),

        # Dengue Visualizations
        dbc.Row([
            dbc.Col(dcc.Graph(id='age-gender-distribution-dengue', figure=dengue_age_gender_distribution(), 
                              style={'height': '400px'}), width=6),
            dbc.Col(dcc.Graph(id='cases-over-time-dengue', figure=dengue_cases_over_time(), 
                              style={'height': '400px'}), width=6),
        ], className="mb-4", style={'margin-bottom': '30px'}),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='hemorrhagic-cases-dengue', figure=dengue_hemorrhagic_cases_pie(), 
                              style={'height': '400px'}), width=6),
            dbc.Col(dcc.Graph(id='comorbidity-heatmap-dengue', figure=dengue_comorbidity_heatmap(),
                              style={'height': '400px'}), width=6),
        ], className="mb-4", style={'margin-bottom': '30px'}),
        
        
        # Febriles Subtitle
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        html.H4("Diseases with Fever and Exanthema Dataset", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section presents exploratory data analysis (EDA) for the Diseases with Fever and Exanthema dataset, "
            "covering demographic, categorical, and geospatial insights.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),

        # Febriles Visualizations
        dbc.Row([
            dbc.Col(dcc.Graph(id='febriles-strip', figure=febriles_strip_plot(), 
                              style={'height': '500px'}), width=6),
            dbc.Col(dcc.Graph(id='febriles-parallel-categories', figure=febriles_parallel_categories(), 
                            style={'height': '500px'}), width=6),
        ], className="mb-4", style={'margin-bottom': '15px'}),

        dbc.Row([
            dbc.Col(dcc.Graph(id='febriles-geospatial', figure=febriles_geospatial_distribution(), 
                            style={'height': '800px', 'width': '100%'}), width=10),
        ], className="justify-content-center mb-4", style={'margin-bottom': '30px'}),


        # Morbilidad Subtitle
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        html.H4("Morbilidad Dataset", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section presents exploratory data analysis (EDA) for the Morbilidad dataset, "
            "highlighting trends, proportions, and geospatial distributions of diseases.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),

        # Morbilidad Visualizations
        dbc.Row([
            dbc.Col(dcc.Graph(id='morbilidad-top-diseases', figure=morbilidad_top_diseases_over_time(), 
                            style={'height': '500px'}), width=12),
        ], className="mb-4", style={'margin-bottom': '30px'}),

        dbc.Row([
            dbc.Col(dcc.Graph(id='morbilidad-treemap', figure=morbilidad_treemap(), 
                            style={'height': '800px'}), width=11),
        ], className="mb-4", style={'margin-bottom': '15px'}),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='morbilidad-geospatial', figure=morbilidad_geospatial_distribution(), 
                            style={'height': '800px'}), width=10),
        ], className="justify-content-center mb-4", style={'margin-bottom': '30px'}),
    ]
)


# Generate results
covid_dengue_manova_summary = manova_covid_dengue()
covid_single_manova_summary = manova_covid_single()

# Add tables and plots to the dashboard
stat_tests_tab = dbc.Tab(
    label="Statistical Tests",
    children=[
        # ANOVA Section
        html.H4("Analysis of Variance (ANOVA)", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section presents ANOVA results analyzing differences in age across the datasets (COVID, Dengue, Febriles).",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='anova-age-datasets', figure=anova_age_across_datasets(), 
                              style={'height': '500px'}), width=11),
        ], className="justify-content-center mb-4", style={'margin-bottom': '30px'}),

        # Multiple Datasets MANOVA Section
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        html.H4("Multivariate Analysis of Variance (MANOVA)", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P([
            "This section presents MANOVA results for analyzing differences in comorbidities ",
            "(Diabetes, Hypertension, and Pregnancy) between COVID and Dengue datasets.",
            html.Br(),
            "This section also presents MANOVA results for analyzing differences in comorbidities ",
            "(Diabetes, Hypertension, Asthma, EPOC, and Obesity) between ambulatory and hospitalized patients in the COVID dataset.",
        ], style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Results: COVID vs. Dengue", style={'text-align': 'center', 'margin-bottom': '20px'}),
                dash_table.DataTable(
                    id='manova-covid-dengue-summary',
                    columns=[{"name": col, "id": col} for col in covid_dengue_manova_summary.columns],
                    data=covid_dengue_manova_summary.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_as_list_view=True,
                )
            ]), width=10),
        ], className="justify-content-center mb-4", style={'margin-bottom': '40px'}),

        # Single Dataset MANOVA Section
        html.Br(),
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Results: Ambulatory vs. Hospitalized COVID Patients", style={'text-align': 'center', 'margin-bottom': '20px'}),
                dash_table.DataTable(
                    id='manova-covid-single-summary',
                    columns=[{"name": col, "id": col} for col in covid_single_manova_summary.columns],
                    data=covid_single_manova_summary.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_as_list_view=True,
                )
            ]), width=10),
        ], className="justify-content-center mb-4", style={'margin-bottom': '50px'}),
    ]
)

clustering_tab = dbc.Tab(
    label="Clustering",
    children=[
        # Agglomerative and Time Series Clustering Section
        html.H4("Agglomerative and Time Series Clustering", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section presents clustering analyses, including K-Means and K-Prototypes for grouping based on "
            "numerical and mixed data, as well as hierarchical clustering and time-series clustering for uncovering "
            "temporal patterns and structural groupings.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='kmeans-dengue', figure=kmeans_dengue_clustering(), style={'height': '500px'}), width=10),
        ], className="justify-content-center mb-4", style={'margin-bottom': '30px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='kprototypes-febriles', figure=kprototypes_febriles(), style={'height': '500px'}), width=10),
        ], className="justify-content-center mb-4", style={'margin-bottom': '40px'}),
        dbc.Row([
            dbc.Col(html.Div(hierarchical_clustering_covid(), style={'height': '500px'}), width=5),
            dbc.Col(dcc.Graph(id='time-series-clustering-means', figure=time_series_clustering_morbidities(), style={'height': '500px'}), width=6),
        ], className="justify-content-center mb-4", style={'margin-bottom': '40px'}),

        # PCA Section
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px', 'margin-top': '20px'}),
        html.H4("Principal Component Analysis (PCA)", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "Principal Component Analysis is used for dimensionality reduction, allowing us to represent complex datasets "
            "in a simplified form by extracting the components that explain the most variance in the data. The 2D and 3D PCA "
            "plots below show the distribution of states and their disease profiles based on their principal components.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='pca-covid', figure=pca_covid(), style={'height': '500px'}), width=10),
        ], className="justify-content-center mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='pca-morbidities-3d', figure=pca_morbidities_3d(), style={'height': '700px'}), width=12),
        ], className="justify-content-center mb-4"),

        # Factor Analysis Section
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        html.H4("Factor Analysis (FA)", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "Factor Analysis is a statistical method used to identify latent factors that explain the variability in a dataset. "
            "By reducing the dimensionality of the data, FA helps uncover patterns and groupings of related variables. Below, "
            "we explore the relationships between comorbidities in the COVID, Dengue, and Febriles datasets.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='fa-covid', figure=factor_analysis_covid(), style={'height': '500px'}), width=6),
            dbc.Col(dcc.Graph(id='fa-dengue', figure=factor_analysis_dengue(), style={'height': '500px'}), width=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='fa-febriles', figure=factor_analysis_febriles(), style={'height': '500px'}), width=6),
        ], className="justify-content-center mb-4"),
    ]
)

# Generate results
logistic_model, logistic_acc, logistic_report, logistic_cm, logistic_feature_importance = logistic_regression_tipo_paciente()
rf_model, rf_acc, rf_report, rf_cm, rf_feature_importance, X_test_rf, y_test_rf, y_pred_proba_rf = random_forest_clasificacion_final()
rf_precision_recall_curve = precision_recall_curve_rf(y_test_rf, y_pred_proba_rf, n_classes=7)

rf_d_model, rf_d_acc, rf_d_report, rf_d_cm, rf_d_feature_importance, X_test_rf_d, y_test_rf_d, y_pred_proba_rf_d = random_forest_dictamen()
qda_model, qda_acc, qda_report, qda_cm, qda_feature_importance = qda_classification_dengue()
gb_acc, gb_report, gb_cm, gb_feature_importance = load_gradient_boosting_results()


# Extract weighted metrics for each model
logistic_metrics_df = extract_weighted_metrics(logistic_report, logistic_acc).round(3)
rf_metrics_df = extract_weighted_metrics(rf_report, rf_acc).round(3)

rf_d_metrics_df = extract_weighted_metrics(rf_d_report, rf_d_acc).round(3)
qda_metrics_df = extract_weighted_metrics(qda_report, qda_acc).round(3)
gb_metrics_df = extract_weighted_metrics(gb_report, gb_acc).round(3)

classification_tab = dbc.Tab(
    label="Classification",
    children=[
        # COVID Section
        html.H4("COVID-19 Classification Results", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section showcases the results of classification models applied to the COVID-19 dataset. "
            "We include Logistic Regression and Random Forest methods to predict patient types and COVID classifications.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),

        # Logistic Regression Results
        html.H5("Logistic Regression for TIPO_PACIENTE", style={'text-align': 'center', 'margin-top': '20px'}),
        dbc.Row([
            # Metrics Table
            dbc.Col(html.Div([
                html.H6("Metrics Summary", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='logistic-metrics-table',
                    columns=[{'name': col, 'id': col} for col in logistic_metrics_df.columns],
                    data=logistic_metrics_df.to_dict('records'),
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ], style={'align-items': 'center', 'justify-content': 'center'}), width=5),
        
            # Feature Importance
            dbc.Col(dcc.Graph(
                id='logistic-feature-importance',
                figure=px.bar(
                    logistic_feature_importance, 
                    x='Feature', y='Coefficient',
                    title="Feature Importance (Logistic Regression)",
                    template='seaborn',
                    color='Feature',
                    color_discrete_sequence=['#006EC1', '#009EE5', '#89D1F3']
                )
            ), width=6),
        ], className="mb-4 justify-content-center", style={'display': 'flex', 'align-items': 'center'}),

        # Confusion Matrix
        html.Br(),
        dbc.Row([
            dbc.Col(html.Div([
                html.H6("Confusion Matrix", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='logistic-confusion-matrix',
                    columns=[
                        {'name': 'Predicted Ambulatorio', 'id': 'Ambulatorio'},
                        {'name': 'Predicted Hospitalizado', 'id': 'Hospitalizado'}
                    ],
                    data=[
                        {"Ambulatorio": logistic_cm[0][0], "Hospitalizado": logistic_cm[0][1]},
                        {"Ambulatorio": logistic_cm[1][0], "Hospitalizado": logistic_cm[1][1]}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ]), width=6),
        ], className="mb-4 justify-content-center", style={'margin-bottom': '100px'}),

        # Random Forest Results
        html.H5("Random Forest for CLASIFICACION_FINAL", style={'text-align': 'center', 'margin-top': '100px'}),
        dbc.Row([
            # Metrics Table
            dbc.Col(html.Div([
                html.H6("Metrics Summary", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='rf-metrics-table',
                    columns=[{'name': col, 'id': col} for col in rf_metrics_df.columns],
                    data=rf_metrics_df.to_dict('records'),
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ], style={'align-items': 'center', 'justify-content': 'center'}), width=5),
            
            # Precision-Recall Curve
            dbc.Col(dcc.Graph(
                id='rf-precision-recall',
                figure=rf_precision_recall_curve, 
                style={'height': '400px'}
            ), width=6),
        ], className="mb-4 justify-content-center", style={'display': 'flex', 'align-items': 'center'}),

        # Confusion Matrix
        dbc.Row([
            dbc.Col(html.Div([
                html.H6("Confusion Matrix", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='rf-confusion-matrix',
                    columns=[
                        {'name': f'Predicted {cls}', 'id': f'Class {i}'}
                        for i, cls in enumerate([
                            'Association', 'Decease', 'Laboratory', 'Invalid', 'Not Applicable', 'Suspected', 'Negative'
                        ])
                    ],
                    data=[
                        {f'Class {i}': row[i] for i in range(len(row))}
                        for row in rf_cm  # Confusion matrix
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ]), width=12),
        ], className="mb-4 justify-content-center", style={'margin-bottom': '150px'}),
        
        
        # Dengue Section
        html.Br(style={'margin-bottom': '200px'}),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        html.H4("Dengue Classification Results", style={'text-align': 'center', 'margin-top': '20px'}),
        html.P(
            "This section showcases the results of classification models applied to the Dengue dataset. "
            "We include Gradient Boosting, Random Forest, and QDA methods to predict various outcomes, such as PCR results, diagnosis, and mortality.",
            style={'text-align': 'center', 'margin-bottom': '40px'}
        ),
        html.Hr(style={'border': '1px solid #ccc', 'margin-bottom': '30px'}),
        
        
        # Gradient Boosting Results
        html.H5("Gradient Boosting for RESULTADO_PCR", style={'text-align': 'center', 'margin-top': '20px'}),
        dbc.Row([
            # Metrics Table
            dbc.Col(html.Div([
                html.H6("Metrics Summary", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='gb-metrics-table',
                    columns=[{'name': col, 'id': col} for col in gb_metrics_df.columns],
                    data=gb_metrics_df.to_dict('records'),
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ], style={'align-items': 'center', 'justify-content': 'center'}), width=5),

            # Feature Importance
            dbc.Col(dcc.Graph(
                id='gb-feature-importance',
                figure=px.bar(
                    gb_feature_importance,
                    x='Feature', y='Importance',
                    title="Feature Importance (Gradient Boosting)",
                    template='seaborn',
                    color='Feature',
                    color_discrete_sequence=['#006EC1', '#009EE5', '#52BCEC', '#89D1F3', '#B5E5F9']
                )
            ), width=6),
        ], className="mb-4 justify-content-center", style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '30px'}),       
        
        
        # Random Forest Results
        html.Br(style={'margin-bottom': '100px'}),
        html.H5("Random Forest for DICTAMEN", style={'text-align': 'center', 'margin-top': '20px'}),
        dbc.Row([
            # Metrics Table
            dbc.Col(html.Div([
                html.H6("Metrics Summary", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='rf-d-metrics-table',
                    columns=[{'name': col, 'id': col} for col in rf_d_metrics_df.columns],
                    data=rf_d_metrics_df.to_dict('records'),
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ], style={'align-items': 'center', 'justify-content': 'center'}), width=5),

            # ROC Curve
            dbc.Col(dcc.Graph(
                id='rf-d-roc-curve',
                figure=roc_curve_rf(y_test_rf_d, y_pred_proba_rf_d),
                style={'height': '400px'}
            ), width=6),
        ], className="mb-4 justify-content-center", style={'display': 'flex', 'align-items': 'center'}),

        # Confusion Matrix
        dbc.Row([
            dbc.Col(html.Div([
                html.H6("Confusion Matrix", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='rf-d-confusion-matrix',
                    columns=[
                        {'name': f'Predicted {cls}', 'id': f'Class {i}'}
                        for i, cls in enumerate(['Dengue', 'Chikungunya', 'Negative'])  # Limit to three columns
                    ],
                    data=[
                        {f'Class {i}': row[i] if i < len(row) else None for i in range(3)}  # Ensure only 3 columns are passed
                        for row in rf_d_cm
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ]), width=6),
        ], className="mb-4 justify-content-center", style={'margin-bottom': '100px'}),


        # QDA Results
        html.H5("Quadratic Discriminant Analysis (QDA) for DEFUNCION", style={'text-align': 'center', 'margin-top': '100px'}),
        dbc.Row([
            # Metrics Table
            dbc.Col(html.Div([
                html.H6("Metrics Summary", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='qda-metrics-table',
                    columns=[{'name': col, 'id': col} for col in qda_metrics_df.columns],
                    data=qda_metrics_df.to_dict('records'),
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ], style={'align-items': 'center', 'justify-content': 'center'}), width=5),

            # QDA Projection Plot
            dbc.Col(dcc.Graph(
                id='qda-projection',
                figure=qda_dengue_defuncion(),
                style={'height': '400px'}
            ), width=6),
        ], className="mb-4 justify-content-center", style={'display': 'flex', 'align-items': 'center'}),

        # Confusion Matrix
        dbc.Row([
            dbc.Col(html.Div([
                html.H6("Confusion Matrix", style={'text-align': 'center'}),
                dash_table.DataTable(
                    id='qda-confusion-matrix',
                    columns=[
                        {'name': 'Predicted Alive', 'id': 'Alive'},
                        {'name': 'Predicted Deceased', 'id': 'Deceased'}
                    ],
                    data=[
                        {"Alive": qda_cm[0][0], "Deceased": qda_cm[0][1]},
                        {"Alive": qda_cm[1][0], "Deceased": qda_cm[1][1]}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_as_list_view=True
                )
            ]), width=6),
        ], className="mb-4 justify-content-center", style={'margin-bottom': '100px'})
    ]
)





# Create the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Project: Analysis of Morbidity Patterns and Use of Services in the IMSS"

# Combine tabs into layout
app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Project: Analysis of Morbidity Patterns and Use of Services in the IMSS",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        dbc.Tabs([eda_tab, stat_tests_tab, clustering_tab, classification_tab]),
        html.Footer(
            [
                html.Hr(),
                html.Div(
                    [
                        html.P(
                            "Sources of Information:",
                            style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "5px"}
                        ),
                        html.P(
                            [
                                "1. Dirección General de Epidemiología: ",
                                html.A("Anuarios Estadísticos de Morbilidad 1984-2023", href="https://epidemiologia.salud.gob.mx/anuario/html/morbilidad_grupo.html", target="_blank"),
                            ],
                            style={"fontSize": "14px", "marginBottom": "10px"}
                        ),
                        html.P(
                            [
                                "2. Gobierno de México: ",
                                html.A("Datos Abiertos", href="https://www.gob.mx/salud/documentos/datos-abiertos-152127", target="_blank"),
                            ],
                            style={"fontSize": "14px", "marginBottom": "10px"}
                        ),
                        html.P(
                            "Data collected under sentinel surveillance methods as recommended by WHO, ensuring sample representativity through 475 USMER units across Mexico.",
                            style={"fontSize": "14px", "marginBottom": "10px"}
                        ),
                        html.P(
                            "Data is provided under the regulations of the Open Data Decree published in the Official Gazette of the Federation on February 20, 2015.",
                            style={"fontSize": "14px"}
                        ),
                    ],
                    style={"textAlign": "center", "marginTop": "20px"}
                ),
            ],
            style={"backgroundColor": "#f8f9fa", "padding": "20px"}
        )
    ],
    fluid=True,
    style={"padding": "20px"}
)

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host='127.0.0.1', port=port)