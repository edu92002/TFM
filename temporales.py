import pandas as pd
from tsfeatures import tsfeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import  cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import ast
from datetime import timedelta
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text,plot_tree
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor


def visinter(model,X,y,conteo_interacciones):
    # 1. Obtener predicciones para todo el conjunto de datos
    y_pred = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]  # Probabilidades de clase positiva

    # 2. Calcular el número de interacciones por instancia (ajusta según tu estructura)
    # Asumiendo que cada fila en X contiene una serie temporal:

    print(X)
    print(conteo_interacciones)
    # 3. Crear el gráfico
    plt.figure(figsize=(12, 6))
    # Gráfico de dispersión con colores según clase real
    scatter = plt.scatter(conteo_interacciones.iloc[:,1], y_pred, c=y, cmap='coolwarm', alpha=0.6, 
                        edgecolors='w', linewidth=0.5)

    # Línea de decisión (umbral 0.5)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)

    # Personalización
    plt.colorbar(scatter, label='Clase Real (0=Humano, 1=Bot)')
    plt.xlabel('Número total de interacciones')
    plt.ylabel('Probabilidad predicha de ser bot')
    plt.title('Relación entre Interacciones y Clasificación\n(Todos los folds de CV)')

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def imputar(X):
    for col in X.columns:
        n_missing = X[col].isna().mean() * 100  # Multiplica por 100 para obtener porcentaje
        if 0 < n_missing <= 10:
            datos_completos = X.dropna(subset=[col])
            datos_faltantes = X[X[col].isna()]
            # Evitar columnas con todos los valores iguales o sin variabilidad
            if datos_completos[col].nunique() <= 1:
                continue
            # Preparar features (descartar la columna objetivo)
            X_train = datos_completos.drop(columns=[col])
            y_train = datos_completos[col]
            X_pred  = datos_faltantes.drop(columns=[col])
            # Asegurarse de que no hay NaNs en los features de entrenamiento/predicción
            if X_train.isna().any().any() or X_pred.isna().any().any():
                continue  # saltar columnas donde los features tienen NaNs
            modelo = HistGradientBoostingRegressor()
            modelo.fit(X_train, y_train)
            X.loc[X[col].isna(), col] = modelo.predict(X_pred)
    return X


def main():
    # 1. Cargar y preparar datos
    df = pd.read_csv("interacciones_mayores_30_confake.csv",encoding='latin1')
    df['fechas_interacciones'] = df['fechas_interacciones'].apply(ast.literal_eval)
    df = df.explode('fechas_interacciones')
    df['fechas_interacciones'] = pd.to_datetime(df['fechas_interacciones'])
    
    # 2. Crear identificador único por tweet y día
    df['fecha'] = df['fechas_interacciones'].dt.date
    df['intervalo'] = df['fechas_interacciones'].dt.floor('4H').dt.time

    bot_count = df.groupby('original_tweet_id')['bots'].sum().reset_index()
    bot_count['is_bot'] = (bot_count['bots'] > 10).astype(int)
    conteo = bot_count['is_bot'].value_counts()
    porcentaje = bot_count['is_bot'].value_counts(normalize=True) * 100
    
    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'Conteo': conteo,
        'Porcentaje': porcentaje.round(2)
    })
    print(resultados)
    # 3. Agrupar por tweet, fecha e intervalo
    grouped = (
        df.groupby(['original_tweet_id', 'fecha', 'intervalo'])
        .size()
        .reset_index(name='y')
        .rename(columns={'original_tweet_id': 'unique_id'})
    )
    grouped['y'] = grouped['y'].astype(float)

    # 4. Crear series temporales de dos semanas mínimas
    complete_series = []
    for uid, group in grouped.groupby('unique_id'):
        # Determinar fecha de inicio para esta serie
        start_date = group['fecha'].min()
        # ADDED: asegurar ventana de al menos dos semanas (14 días)
        end_date = start_date + timedelta(days=13)
        # Generar todas las marcas horarias en el rango de dos semanas
        ds = pd.date_range(start=start_date,
                           end=end_date + pd.Timedelta(hours=23),
                           freq='4H')
        full_df = pd.DataFrame({'ds': ds})
        full_df['unique_id'] = uid
        full_df['fecha'] = full_df['ds'].dt.date
        full_df['intervalo'] = full_df['ds'].dt.time

        # Combinar con datos reales y rellenar con 0 donde falten
        full_df = full_df.merge(
            group[['fecha', 'intervalo', 'y']],
            on=['fecha', 'intervalo'],
            how='left'
        ).fillna({'y': 0})

        complete_series.append(full_df)

    complete_df = pd.concat(complete_series)
    print(complete_df)
    conteo_interacciones = complete_df.groupby('unique_id')['y'].sum().reset_index()
    # 5. Calcular características por día
    features_df = tsfeatures(
        complete_df,
        freq=1,  
        threads=1,
    ).reset_index()

    # 6. Combinar con variable objetivo
    final_data = features_df.merge(
        bot_count[['original_tweet_id', 'is_bot']],
        left_on='unique_id',
        right_on='original_tweet_id',
        how='inner'
    )

    y = final_data['is_bot']


    # AÑADIDO: combinar texto con X
    X = final_data.drop(['index','unique_id', 'original_tweet_id', 'is_bot'], axis=1)

    X = imputar(X)
    X = X.dropna(axis=1)
    X.columns = X.columns.astype(str)
    print(X)

    # 8. Validación cruzada
    if len(X) > 0:
        # Definición de modelos
        models = {
            'Random Forest': make_pipeline(
                StandardScaler(),
                RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
            ),
            'SVM': make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', class_weight='balanced', random_state=42)
            ),
            'C4.5 (Decision Tree)': make_pipeline(
                StandardScaler(),
                DecisionTreeClassifier(criterion='entropy',
                                        class_weight='balanced', random_state=42, max_depth=3)
            )
        }
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        # Evaluación de cada modelo
        for name, model in models.items():
            print(f"\n=== Evaluación de {name} ===")
            results = cross_validate(model, X, y, cv=5, scoring=scoring)
            
            print(f"Accuracy : {results['test_accuracy'].mean():.2f} ± {results['test_accuracy'].std():.2f}")
            print(f"Precision: {results['test_precision'].mean():.2f} ± {results['test_precision'].std():.2f}")
            print(f"Recall   : {results['test_recall'].mean():.2f} ± {results['test_recall'].std():.2f}")
            print(f"F1 Score : {results['test_f1'].mean():.2f} ± {results['test_f1'].std():.2f}")
            
            # Visualización especial para el árbol de decisión (C4.5)
            if name == 'C4.5 (Decision Tree)':
                # Entrenar el modelo una vez más para visualización
                dt_model = model.fit(X, y)
                dt = dt_model.named_steps['decisiontreeclassifier']
                
                print("\nEstructura del Árbol de Decisión (C4.5):")
                tree_rules = export_text(dt, feature_names=list(X.columns) if hasattr(X, 'columns') else None)
                print(tree_rules)
                
                try:
                    dt = model.named_steps['decisiontreeclassifier']
                
                    plt.figure(figsize=(12, 8))
                    plot_tree(dt, 
                                feature_names=X.columns if hasattr(X, 'columns') else None,
                                class_names=[str(c) for c in dt.classes_],
                                filled=True, rounded=True)
                    plt.title("Árbol de Decisión (C4.5)")
                    plt.show()
                except Exception as e:
                    print(f"No se pudo generar la visualización del árbol: {e}")
    else:
        print("No hay datos válidos para entrenar")

    visinter(model,X,y,conteo_interacciones) #c4.5


if __name__ == "__main__":
    main()




