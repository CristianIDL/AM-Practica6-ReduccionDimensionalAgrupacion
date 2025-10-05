import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, load_wine

# Estandarizar los datos
from sklearn.preprocessing import StandardScaler

# Técnicas de reducción dimensional
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

datasets = {
    "digits": load_digits(),
    "iris": load_iris(),
    "wine": load_wine()
}

tecnicas = {
    "PCA": PCA,
    "ICA": FastICA,
    "t-SNE": TSNE,
    "Random Projection": GaussianRandomProjection
}

def ver_dataset_original(dataset_name):
    dataset = datasets[dataset_name]
    X, y = dataset.data, dataset.target
    
    print(f"\n DATASET ORIGINAL: {dataset_name.upper()}")
    print(f"   • Muestras: {X.shape[0]}")
    print(f"   • Características: {X.shape[1]}")
    print(f"   • Clases: {len(np.unique(y))} {np.unique(y)}")
    
    # Distribución de clases
    unique, counts = np.unique(y, return_counts=True)
    for clase, count in zip(unique, counts):
        print(f"   • Clase {clase}: {count} muestras")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    unique_classes = np.unique(y)
    colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for i, clase in enumerate(unique_classes):
        indices = np.where(y == clase)
        ax.scatter(X[indices, 0],
                X[indices, 1] if X.shape[1] > 1 else X[indices, 0],
                X[indices, 2] if X.shape[1] > 2 else X[indices, 0],
                c=colores[i % len(colores)],
            label=f'Clase {clase}',
            marker='o')

    ax.set_xlabel('Característica 0')
    ax.set_ylabel('Característica 1' if X.shape[1] > 1 else 'Característica 0')
    ax.set_zlabel('Característica 2' if X.shape[1] > 2 else 'Característica 0')

    plt.title(f"Clases del Conjunto de Datos {dataset_name}")
    ax.legend()
    plt.show()

def reducir(dataset_name, tecnica_name):
    dataset = datasets[dataset_name]
    
    X, y = dataset.data, dataset.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tecnica = tecnicas[tecnica_name]
    
    try:
        if tecnica_name in ["PCA", "ICA", "Random Projection"]:
            modelo = tecnica(n_components=3, random_state=42)
            X_reducido = modelo.fit_transform(X_scaled)
        elif tecnica_name == "t-SNE":
            modelo = tecnica(n_components=3, random_state=42, init='random', learning_rate='auto')
            X_reducido = modelo.fit_transform(X_scaled)
        else:
            raise ValueError("Técnica no reconocida")

        print(f"\n REDUCCIÓN DE DIMENSIONALIDAD: {tecnica_name} en {dataset_name.upper()}")
        print(f"   • Muestras: {X_reducido.shape[0]}")
        print(f"   • Características reducidas: {X_reducido.shape[1]}")
        print(f"   • Forma original: {X.shape} → Nueva forma: {X_reducido.shape}")
        
        if tecnica_name == "PCA":
            print(f"   • Varianza explicada: {modelo.explained_variance_ratio_}")
            print(f"   • Varianza total explicada: {sum(modelo.explained_variance_ratio_):.3f}")

        unique, counts = np.unique(y, return_counts=True)
        print(f"   • Distribución de clases:")
        for clase, count in zip(unique, counts):
            print(f"     - Clase {clase}: {count} muestras")

        print(f"   ✅ {tecnica_name} aplicado exitosamente\n")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        unique_classes = np.unique(y)
        colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

        for i, clase in enumerate(unique_classes):
            indices = np.where(y == clase)
            ax.scatter(X_reducido[indices, 0],
                    X_reducido[indices, 1],
                    X_reducido[indices, 2],
                    c=colores[i % len(colores)],
                label=f'Clase {clase}',
                marker='o')

        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')

        plt.title(f"{tecnica_name} aplicado a {dataset_name.upper()}")
        ax.legend()
        plt.show()
        
    except Exception as e:
        print(f"   Error con {tecnica_name} en {dataset_name}: {e}")
        print(f"   Continuando con la siguiente técnica...\n")


def main():
    for nombre in datasets.keys():
        ver_dataset_original(nombre)
        for tecnica in tecnicas.keys():
            reducir(nombre, tecnica)


if __name__ == "__main__":
    main()