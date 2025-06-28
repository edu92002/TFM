import pandas as pd
import os
import numpy as np
# Configuración
dataset_path = 'cresci17'
sub_datasets = {
    'fake_followers': 'bot'
}

# id: índice 0
# user_id: índice 3
# in_reply_to_status_id: índice 5
# retweeted_status_id: índice 8

def procesar_subdataset(folder, label):
    folder_path = os.path.join(dataset_path, folder, f"{folder}.csv")
       # Cargar tweets con todas las interacciones
    tweets = pd.read_csv(
        os.path.join(folder_path, 'tweets.csv'),
        dtype={0: str, 3: str, 5: str, 8: str, 1: str, 4: str, 6: str, 9: str},  # Reemplazamos por los índices},  # Reemplazamos por los índices
        encoding='latin1',header=None, quotechar='"', escapechar='\\',on_bad_lines='skip'
    )
    if folder == 'fake_followers':
        columnas_seleccionadas = [1, 4, 6, 9, -1]
    else:
        columnas_seleccionadas = [0, 3, 5, 8, -3]

    # Seleccionamos esas columnas
    tweets = tweets.iloc[:, columnas_seleccionadas]
    tweets.columns = ['id', 'user_id', 'in_reply_to_status_id', 'retweeted_status_id','date']

    if folder == 'fake_followers':
        tweets['original_tweet_id'] = tweets.apply(
        lambda x: x['in_reply_to_status_id'] if x['in_reply_to_status_id'] not in ["0","1"]
                    else x['retweeted_status_id'] if (x['retweeted_status_id'] not in ["0","1"] and not np.isnan(x['retweeted_status_id']))
                    else x['id'],
        axis=1
    )
    else:
        # Defino original_tweet_id
        tweets['original_tweet_id'] = tweets.apply(
            lambda x: x['in_reply_to_status_id'] if x['in_reply_to_status_id'] not in ["0","1"]
                        else x['retweeted_status_id'] if x['retweeted_status_id'] not in ["0","1"]
                        else x['id'],
            axis=1
        )

    # Identifico si es una interacción
    tweets['es_interaccion'] = tweets['original_tweet_id'] != tweets['id']


    # Cargar usuarios y asignar etiqueta
    users = pd.read_csv(
        os.path.join(folder_path, 'users.csv'),
        dtype={'id': str},  # Usamos el índice 0 para 'id'
        usecols=['id']
    )
    users['label'] = label  # Todos los usuarios en esta carpeta tienen la misma etiqueta

    # Filtrar interacciones
    interacciones = tweets[tweets['es_interaccion']]

    # Unir interacciones con usuarios
    merged = pd.merge(
        interacciones, users,
        left_on='user_id', right_on='id',  # 'user_id' y 'id' respectivamente
        how='left'
    )
    print(merged[1:])

    # Agrupar por tweet original
    stats = merged.groupby('original_tweet_id').agg(
        total_interacciones=('user_id', 'count'),
        respuestas=('in_reply_to_status_id', lambda x: (x != "0").sum()),
        retweets=('retweeted_status_id', lambda x: (x != "0").sum()),
        bots=('label', lambda x: (x == 'bot').sum()),
        humanos=('label', lambda x: (x == 'human').sum()),
        tweets_interactuando=('id_x',list),
        usuarios_interactuando=('user_id',list),  # unificamos y quitamos duplicados
        fechas_interacciones=('date',list)
    ).reset_index()

    # Filtrar tweets raíz
    root_tweets = tweets[~tweets['es_interaccion']][['id','user_id']].rename(
        columns={'id':'original_tweet_id', 'user_id':'author_id'}
    )

    print(f"✅ {folder}: {len(interacciones)} interacciones | {len(root_tweets)} tweets raíz")
    return stats, root_tweets

# Listas para acumular
stats_list = []
roots_list = []

for folder, label in sub_datasets.items():
    stats_df, roots_df = procesar_subdataset(folder, label)
    stats_list.append(stats_df)
    roots_list.append(roots_df)

# Concatenamos todas las stats y re-agrupamos para el global
all_stats = pd.concat(stats_list, ignore_index=True)
global_stats = all_stats.groupby('original_tweet_id').agg(
    total_interacciones=('total_interacciones', 'sum'),
    respuestas=('respuestas', 'sum'),
    retweets=('retweets', 'sum'),
    bots=('bots', 'sum'),
    humanos=('humanos', 'sum'),
    tweets_interactuando=('tweets_interactuando',lambda lists: 
        list({uid for lst in lists for uid in lst})),  # unificamos y quitamos duplicados),
    usuarios_interactuando=('usuarios_interactuando', lambda lists: 
        list({uid for lst in lists for uid in lst})  # unificamos y quitamos duplicados
    ),
    fechas_interacciones=('fechas_interacciones', lambda lists: 
        list({uid for lst in lists for uid in lst})  # unificamos y quitamos duplicados
    )
).reset_index()

# Para los root tweets, basta concatenar y quitar duplicados
all_roots = pd.concat(roots_list, ignore_index=True)
global_roots = all_roots.drop_duplicates(subset='original_tweet_id')

# Creamos carpeta de resultados globales
os.makedirs('resultados/global', exist_ok=True)
global_stats.to_csv('resultados/global/pruebifake.csv', index=False)
global_roots.to_csv('resultados/global/rootsprebifake.csv', index=False)

print("\n✅ Cálculo global completado.")
