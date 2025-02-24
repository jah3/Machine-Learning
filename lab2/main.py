import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datele privind performanța modelelor de clasificare
model_metrics = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [0.945, 0.92, 0.93],
    'Precision': [0.55, 0.60, 0.65],
    'Recall': [0.17, 0.40, 0.50],
    'F1-Score': [0.26, 0.48, 0.55]
})

# Configurarea paginii Streamlit
st.set_page_config(page_title="Compararea Performanței Modelelor", layout="wide")

# Titlul aplicației
st.title("Evaluarea Performanței Modelelor de Clasificare")

# Afișarea tabelului de rezultate
st.subheader("Măsurători de Performanță")
st.dataframe(model_metrics)

# Crearea vizualizărilor
st.subheader("Compararea Metricilor")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=model_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Metric', y='Score', hue='Model', palette="viridis", ax=ax)
ax.set_title("Compararea Performanțelor Modelelor")
ax.set_ylabel("Valoare metrică")
ax.set_xlabel("Metrică")
st.pyplot(fig)
