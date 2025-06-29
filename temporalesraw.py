import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import ast
from datetime import timedelta
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text,plot_tree
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict


def visbarras(plot_data):
    # Seleccionar 1 bot y 1 humano aleatorios
    sample_bot = plot_data[plot_data['is_bot'] == 1]['unique_id'].sample(1).values[0]
    sample_human = plot_data[plot_data['is_bot'] == 0]['unique_id'].sample(1).values[0]
    # --- Gráfico del BOT ---
    plt.figure(figsize=(15, 5))
    bot_data = plot_data[plot_data['unique_id'] == sample_bot]
    print(bot_data)
    plt.bar(bot_data['t'],bot_data['y'], color='red', alpha=0.7, width=0.8)
    
    plt.title(f'Patrón de Actividad Temporal: Bot (ID: {sample_bot})', fontsize=14)
    plt.xlabel('Unidad temporal', fontsize=12)
    plt.ylabel('Número de Interacciones', fontsize=12)
    plt.xticks(range(0, 336),fontsize=5)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(-0.5,100)
    plt.tight_layout()
    plt.show()
    
    # --- Gráfico del HUMANO ---
    plt.figure(figsize=(15, 5))
    human_data = plot_data[plot_data['unique_id'] == sample_human]
    plt.bar(human_data['t'],human_data['y'], color='blue', alpha=0.7, width=0.8)
    
    plt.title(f'Patrón de Actividad Temporal: Humano (ID: {sample_human})', fontsize=14)
    plt.xlabel('Unidad temporal', fontsize=12)
    plt.ylabel('Número de Interacciones', fontsize=12)
    plt.xticks(range(0,336),fontsize=5)
    plt.xlim(-0.5,100)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def visinter(model,X,y):
    # 1. Obtener predicciones para todo el conjunto de datos
    y_pred = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]  # Probabilidades de clase positiva

    # 2. Calcular el número de interacciones por instancia (ajusta según tu estructura)
    # Asumiendo que cada fila en X contiene una serie temporal:
    print(X)
    num_interacciones = X.sum(axis=1)  # Suma a lo largo del eje temporal
    print(num_interacciones)
    # 3. Crear el gráfico
    plt.figure(figsize=(12, 6))

    # Gráfico de dispersión con colores según clase real
    scatter = plt.scatter(num_interacciones, y_pred, c=y, cmap='coolwarm', alpha=0.6, 
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

def main():
    # 1. Cargar y preparar datos
    df = pd.read_csv("interacciones_mayores_30_confake.csv")
    df['fechas_interacciones'] = df['fechas_interacciones'].apply(ast.literal_eval)
    df = df.explode('fechas_interacciones')
    df['fechas_interacciones'] = pd.to_datetime(df['fechas_interacciones'])
    
    # 2. Crear identificador único por tweet y día
    df['fecha'] = df['fechas_interacciones'].dt.date
    df['intervalo'] = df['fechas_interacciones'].dt.floor('1H').dt.time

    bot_count = df.groupby('original_tweet_id')['bots'].sum().reset_index()
    bot_count['is_bot'] = (bot_count['bots'] > 10).astype(int)  # Más de 10 bots = 1

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
        start_date = group['fecha'].min()
        end_date = start_date + timedelta(days=13)
        # Marks each hour over two weeks (336 hours)
        ds = pd.date_range(start=start_date,
                           end=end_date + pd.Timedelta(hours=24),
                           freq='1H')
        full_df = pd.DataFrame({'ds': ds})
        full_df['unique_id'] = uid
        full_df['fecha'] = full_df['ds'].dt.date
        full_df['intervalo'] = full_df['ds'].dt.time

        # Merge con datos reales y rellenar ceros donde falte
        full_df = full_df.merge(
            group[['fecha', 'intervalo', 'y']],
            on=['fecha', 'intervalo'],
            how='left'
        ).fillna({'y': 0})

        complete_series.append(full_df)

    complete_df = pd.concat(complete_series)

    # 5. Transformar a formato ancho: una fila por tuit, 336 columnas de horas
    # Índice temporal relativo en horas desde el inicio de cada serie
    complete_df['t'] = ((complete_df['ds'] - complete_df.groupby('unique_id')['ds']
                         .transform('min'))
                        .dt.total_seconds() // 3600).astype(int)
    wide_df = complete_df.pivot(index='unique_id', columns='t', values='y')
    wide_df = wide_df.fillna(0)
    # Renombrar columnas a h_0 ... h_335
    wide_df.columns = [f'h_{int(col)}' for col in wide_df.columns]

    # 6. Combinar con variable objetivo y preparar X, y
    final_df = wide_df.reset_index().merge(
        bot_count[['original_tweet_id', 'is_bot']],
        left_on='unique_id',
        right_on='original_tweet_id',
        how='inner'
    )

    print(final_df)
    X = final_df.drop(['unique_id', 'original_tweet_id', 'is_bot'], axis=1)
    y = final_df['is_bot']

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
                    DecisionTreeClassifier(criterion='entropy', max_depth=3, 
                                            class_weight='balanced', random_state=42)
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

                # y_pred = cross_val_predict(model, X, y, cv=5)

                # # Matriz de confusión
                # cm = confusion_matrix(y, y_pred)
                # print("Matriz de confusión:")
                # print(cm)
                
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
# --- Código para visualización de series en dos gráficos ---

    visinter(model,X,y) #c4.5
    if len(X) > 0:
        # Preparar datos para plotting
        plot_data = complete_df.merge(
            final_df[['unique_id', 'is_bot']],
            on='unique_id'
        )
        visbarras(plot_data)

if __name__ == "__main__":
    main()

