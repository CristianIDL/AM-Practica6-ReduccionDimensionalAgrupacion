from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, Birch, AgglomerativeClustering
from sklearn.datasets import load_iris, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 42

datasets = {
    "iris": load_iris(),
    "blobs": make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=RANDOM_STATE),
    "moons": make_moons(n_samples=300, noise=0.05, random_state=RANDOM_STATE),
    "mall_customers": pd.read_csv('Mall_Customers.csv')
}

# Preprocesamiento específico para Mall Customers
def prepare_mall_customers(df):
    """Prepara el dataset de Mall Customers para clustering"""
    # Usamos las columnas típicas para segmentación de clientes
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    # Estandarizamos porque las escalas son diferentes
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Si tienes el dataset de mall customers, aplica la preparación
if 'mall_customers' in datasets:
    datasets['mall_customers_data'] = prepare_mall_customers(datasets['mall_customers'])

configuraciones = {
    "iris" : {
        "n_clusters_guess": 3, # Número esperado de clusters
        "dbscan_eps": 0.5, # Distancia máxima entre dos muestras para que una sea considerada en el vecindario de la otra
        "dbscan_min_samples": 5, # Número mínimo de muestras en un vecindario para que un punto sea considerado como núcleo
        "true_labels": datasets["iris"].target,
        "dimensions": 2
    },
    "blobs": {
        "n_clusters_guess": 4,
        "dbscan_eps": 0.3,
        "dbscan_min_samples": 5,
        "true_labels": datasets["blobs"][1],
        "dimensions": 2
    },
    "moons": {
        "n_clusters_guess": 2,
        "dbscan_eps": 0.2,  # Lo hacemos más pequeño para que se ven las dos lunas
        "dbscan_min_samples": 5,
        "true_labels": datasets["moons"][1],
        "dimensions": 2
    },
    "mall_customers_data": {
        "n_clusters_guess": 5,
        "dbscan_eps": 0.4,
        "dbscan_min_samples": 5,
        "true_labels": None,  # No tenemos etiquetas verdaderas
        "dimensions": 3
    }
}

# Función para aplicar clustering
def aplicar_clustering(X, dataset_name):
    # Aplicamos todos los algoritmos de clustering

    config = configuraciones[dataset_name]
    n_clusters = config["n_clusters_guess"]

    resultados = {}

    # 1: K-Means
    resultados['KMeans'] = KMeans(
        n_clusters = n_clusters,
        random_state = RANDOM_STATE,
        n_init = 10
    ).fit_predict(X)

    # 2: Spectral Clustering
    resultados['SpectralClustering'] = SpectralClustering(
        n_clusters = n_clusters,
        random_state = RANDOM_STATE,
        affinity = 'nearest_neighbors',
        n_neighbors = 10
    ).fit_predict(X)

    # 3: DBSCAN
    resultados['DBSCAN'] = DBSCAN(
        eps = config["dbscan_eps"],
        min_samples = config["dbscan_min_samples"]
    ).fit_predict(X)

    # 4: Birch
    resultados['Birch'] = Birch(
        n_clusters = n_clusters,
        threshold = 0.5,
        branching_factor = 50
    ).fit_predict(X)

    # 5: Agglomerative Clustering
    resultados['AgglomerativeClustering'] = AgglomerativeClustering(
        n_clusters = n_clusters,
        linkage = 'ward'
    ).fit_predict(X)    

    return resultados

# VISUALIZACIÓN MEJORADA PARA 3D
def visualizar_resultados(X, resultados, dataset_name, true_labels=None):
    """Visualiza los resultados de clustering"""
    
    n_algoritmos = len(resultados)
    dimensions = configuraciones[dataset_name]["dimensions"]
    
    if dimensions == 3:
        # VISUALIZACIÓN 3D
        fig = plt.figure(figsize=(20, 12))
        
        # Datos originales
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], 
                             c=true_labels if true_labels is not None else 'blue',
                             cmap='tab10', s=50, alpha=0.7)
        ax1.set_title(f'{dataset_name} - Datos Originales')
        ax1.set_xlabel('Age (estandarizado)')
        ax1.set_ylabel('Income (estandarizado)')
        ax1.set_zlabel('Spending Score (estandarizado)')
        if true_labels is not None:
            plt.colorbar(scatter, ax=ax1)
        
        # Resultados de cada algoritmo
        algorithms = list(resultados.items())
        for i, (algo_name, labels) in enumerate(algorithms):
            ax = fig.add_subplot(2, 3, i+2, projection='3d')
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                                c=labels, cmap='tab10', s=50, alpha=0.7)
            ax.set_title(f'{algo_name}\nClusters: {len(np.unique(labels))}')
            ax.set_xlabel('Age')
            ax.set_ylabel('Income')
            ax.set_zlabel('Spending Score')
            plt.colorbar(scatter, ax=ax)
        
    else:
        # VISUALIZACIÓN 2D (para los otros datasets)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Usar solo 2D para visualización
        X_plot = X[:, :2] if X.shape[1] > 2 else X
        
        # Datos originales
        if true_labels is not None and hasattr(true_labels, '__len__'):
            scatter = axes[0].scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, cmap='tab10', s=50, alpha=0.7)
            axes[0].set_title(f'{dataset_name} - Etiquetas Verdaderas')
            plt.colorbar(scatter, ax=axes[0])
        else:
            # ¡CORREGIDO: agregar color uniforme para datos sin etiquetas!
            scatter = axes[0].scatter(X_plot[:, 0], X_plot[:, 1], c='blue', s=50, alpha=0.7)
            axes[0].set_title(f'{dataset_name} - Datos Originales')
        
        # Resultados de clustering
        for i, (algo_name, labels) in enumerate(resultados.items()):
            scatter = axes[i+1].scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
            axes[i+1].set_title(f'{algo_name}')
            n_clusters = len(np.unique(labels))
            axes[i+1].set_xlabel(f'Clusters: {n_clusters}')
            plt.colorbar(scatter, ax=axes[i+1])
        
        # Ocultar ejes extras
        for i in range(n_algoritmos + 1, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# EJECUCIÓN PRINCIPAL
def main():
    for dataset_name, data in datasets.items():

        if dataset_name == "mall_customers":
            continue

        print(f"\n{'='*50}")
        print(f"PROCESANDO: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        # Preparar datos
        if dataset_name == "iris":
            X = data.data[:, :2]  # Usamos solo 2 características para visualización
            true_labels = data.target
        elif dataset_name == "mall_customers_data":  # Caso especial para mall_customers_data
            X = data  # Ya está procesado
            true_labels = None
        else:
            X = data if not isinstance(data, tuple) else data[0]
            true_labels = configuraciones[dataset_name]["true_labels"]
        
        # Aplicar clustering
        resultados = aplicar_clustering(X, dataset_name)
        
        # Visualizar
        visualizar_resultados(X, resultados, dataset_name, true_labels)
        
        # Mostrar estadísticas
        print(f"Resumen de clusters encontrados:")
        for algo, labels in resultados.items():
            n_clusters = len(np.unique(labels))
            print(f"  {algo:15}: {n_clusters:2} clusters")

if __name__ == "__main__":
    main()