
# IMSS Morbidity Analysis Dashboard

> 🌐 Live Dashboard: [https://topological-analysis-of-airbnb-price.onrender.com/](https://topological-analysis-of-airbnb-price.onrender.com/)

---

## Overview

This project presents an advanced web-based dashboard developed using Dash, Plotly, and other state-of-the-art Python libraries. The dashboard focuses on analyzing and visualizing epidemiological data from multiple health datasets in Mexico, including COVID-19, Dengue, Febrile Diseases, and Morbidity data. By leveraging powerful statistical techniques, clustering algorithms, and machine learning models, this application provides insights into health trends, comorbidity relationships, and classification tasks related to patient outcomes.

The dashboard integrates interactive visualizations, statistical analysis, and predictive modeling, making it an essential tool for researchers, policymakers, and healthcare providers to understand disease patterns and improve decision-making.

---

## Features

### 1. **Exploratory Data Analysis (EDA)**

- **COVID-19 Dataset:**
  - Age and gender distributions.
  - Temporal trends in COVID-19 case classifications.
  - Geospatial analysis of cases by state.
  - Heatmaps showcasing comorbidity relationships.
  - Analysis of symptoms-to-admission intervals.
- **Dengue Dataset:**
  - Gender-wise age distributions.
  - Trends in dengue cases over time.
  - Proportions of hemorrhagic vs. non-hemorrhagic dengue cases.
  - Comorbidity co-occurrence heatmaps.
- **Febrile Diseases Dataset:**
  - Parallel categories visualization of complications, vaccination, and mortality.
  - Geospatial distribution of cases.
- **Morbidity Dataset:**
  - Temporal trends in the top 10 diseases.
  - Treemap analysis of diseases by state.
  - Geospatial heatmaps of disease prevalence.

### 2. **Statistical Analysis**

- **ANOVA (Analysis of Variance):**
  - Comparing age distributions across datasets (COVID-19, Dengue, and Febrile Diseases).
- **MANOVA (Multivariate Analysis of Variance):**
  - Comparing comorbidities across datasets (e.g., COVID-19 vs. Dengue).
  - Analyzing differences in comorbidities between ambulatory and hospitalized patients.

### 3. **Clustering Analysis**

- **K-Means Clustering:**
  - Applied to numerical variables to group dengue cases.
- **K-Prototypes Clustering:**
  - Applied to mixed-type data (numerical and categorical) in febrile diseases.
- **Hierarchical Clustering:**
  - Applied to COVID-19 data using Ward’s method.
- **Time-Series Clustering:**
  - Clustering morbidity data to uncover temporal patterns.
- **Principal Component Analysis (PCA):**
  - Dimensionality reduction applied to COVID-19 and morbidity datasets.

### 4. **Classification Models**

- **Logistic Regression:**
  - Predicting patient type (ambulatory vs. hospitalized) based on comorbidities in the COVID-19 dataset.
- **Gradient Boosting:**
  - Predicting PCR test results in dengue cases.
- **Quadratic Discriminant Analysis (QDA):**
  - Predicting mortality in dengue cases using demographic and medical data.

---

## Dependencies

This project leverages several advanced libraries to ensure optimal functionality and performance:

- **Data Manipulation:**
  - `pandas`, `numpy`
- **Database & Utils:**
  - `sqlite3` (data storage)
- **Visualization:**
  - `plotly`, `matplotlib`, `dash`, `dash-bootstrap-components`
- **Geospatial Analysis:**
  - `geopandas`, `json`
- **Statistical Analysis:**
  - `scipy`, `statsmodels`
- **Clustering:**
  - `sklearn`, `fastcluster`, `tslearn`, `kmodes`
- **Dimensionality Reduction:**
  - `factor-analyzer`, `scikit-learn`
- **Machine Learning:**
  - `scikit-learn`, `statsmodels`
- **Web Application Framework:**
  - `Dash`, `Plotly`
- **Deployment:**
  - `Render`

---

## Data Sources

1. **Dirección General de Epidemiología:**
   - [Anuarios Estadísticos de Morbilidad (1984-2023)](https://epidemiologia.salud.gob.mx/anuario/html/morbilidad_grupo.html)
2. **Gobierno de México:**
   - [Datos Abiertos](https://www.gob.mx/salud/documentos/datos-abiertos-152127)

Data was collected under sentinel surveillance methods recommended by the WHO, ensuring representativity through 475 USMER units across Mexico. Data usage complies with the Open Data Decree published in the Official Gazette of the Federation on February 20, 2015.

---

## Applications

- **Healthcare Decision Support:**
  - Identifying comorbidity patterns and risk factors for diseases.
  - Evaluating trends in disease prevalence and outcomes over time.
- **Research and Epidemiological Analysis:**
  - Exploring relationships between demographic, medical, and geographical variables.
  - Investigating disease classifications and patient outcomes.
- **Policy Making:**
  - Informing public health interventions based on morbidity and mortality trends.
  - Allocating resources to regions or diseases with the highest burden.

---

## Deployment

The dashboard is deployed on [Render](https://render.com/), allowing public access to all visualizations and results via the following link:
  👉 **[https://topological-analysis-of-airbnb-price.onrender.com/](https://topological-analysis-of-airbnb-price.onrender.com/)**

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bacaSantiago/IMSS-Morbidity-Patterns-Analysis-Dashboard
   cd IMMS-morbidity-patterns-analysis
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare the data:**

   - Place `IMMS_Mexico.sqlite` in the project root (contains Covid, Dengue, Febriles, Morbilidad).
   - Ensure asset files are in `assets/`.

4. **Run the dashboard:**

   ```bash
   python dashboard.py
   ```

   Navigate to `http://127.0.0.1:8050` in your browser.

---

## Future Enhancements

- Integration with real-time data streams for live updates.
- Advanced predictive modeling using deep learning techniques.
- Enhanced geospatial analysis with more granular location data.
- Cross-dataset analysis to identify correlations between diseases.

---

## Author

Developed by **Santiago**, leveraging data science and mathematical expertise to create impactful analytical tools.
