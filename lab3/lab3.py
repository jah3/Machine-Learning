import streamlit as st
import pandas as pd

# Setarea titlului aplicației
st.title("Vizualizare Rezultate Clustering")

# Încărcarea rezultatelor clustering-ului
st.header("Rezultatele Clustering-ului")
clustering_results = pd.read_csv('clustering_results.csv')
st.write(clustering_results)

# Încărcarea rapoartelor de clasificare
st.header("Rapoartele de Clasificare")
classification_results = pd.read_csv('classification_results.csv')
st.write(classification_results)

# Permite utilizatorului să vizualizeze detalii suplimentare
if st.checkbox("Afișează detalii pentru K-Means"):
    st.subheader("Detalii K-Means")
    st.write(classification_results[classification_results['Model'] == 'K-Means'])

if st.checkbox("Afișează detalii pentru DBSCAN"):
    st.subheader("Detalii DBSCAN")
    st.write(classification_results[classification_results['Model'] == 'DBSCAN'])

# Grafic pentru a vizualiza clustering-ul
st.header("Vizualizare Clustering")
if st.button("Vizualizează Clustering"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Încărcăm datele originale pentru vizualizare
    original_data = pd.read_csv('C:\\Users\\user\\Desktop\\ASEM\\ML\\lab2\\bank-full.xls')
    original_data['K-Means'] = clustering_results['Labels'][0]
    original_data['DBSCAN'] = clustering_results['Labels'][1]

    # Crearea unui grafic pentru K-Means
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=original_data.iloc[:, 0], y=original_data.iloc[:, 1], hue=original_data['K-Means'], palette='viridis')
    plt.title('Clustering K-Means')
    plt.xlabel('Dimensiunea 1')
    plt.ylabel('Dimensiunea 2')
    plt.legend(title='Clustere')
    st.pyplot(plt)

    # Crearea unui grafic pentru DBSCAN
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=original_data.iloc[:, 0], y=original_data.iloc[:, 1], hue=original_data['DBSCAN'], palette='viridis')
    plt.title('Clustering DBSCAN')
    plt.xlabel('Dimensiunea 1')
    plt.ylabel('Dimensiunea 2')
    plt.legend(title='Clustere')
    st.pyplot(plt)

# Afișarea unui mesaj final
st.sidebar.title("Opțiuni")
st.sidebar.info("Utilizați aplicația pentru a vizualiza rezultatele clustering-ului și rapoartele de clasificare.")
