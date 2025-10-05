def informacion_dataset(datasets):
    print("=" * 50)
    for name, data in datasets.items():
        print(f"\nDATASET: {name.upper()}")
        print("-" * 30)
        
        if name == "iris":
            print(f"Tipo: {type(data)}")
            print(f"Características: {data.data.shape}")
            print(f"Target: {data.target.shape}")
            print(f"Clases: {np.unique(data.target)}")
            print(f"Nombres clases: {data.target_names}")
            
        elif name == "blobs" or name == "moons":
            X, y = data
            print(f"Tipo X: {type(X)}, Tipo y: {type(y)}")
            print(f"Características: {X.shape}")
            print(f"Target: {y.shape}")
            print(f"Clases: {np.unique(y)}")
            
        elif name == "mall_customers":
            print(f"Tipo: {type(data)}")
            print(f"Dimensiones: {data.shape}")
            print(f"Columnas: {list(data.columns)}")
            print(f"Tipos datos:\n{data.dtypes}")
            print(f"\nPrimeras 3 filas:")
            print(data.head(3))

    print("\n" + "=" * 50)

informacion_dataset()

