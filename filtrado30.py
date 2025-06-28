import pandas as pd

# Leer el CSV forzando la columna 0 como string y evitando el warning
df = pd.read_csv("interacciones_por_tweet_global_texto.csv", encoding="cp1252", dtype={0: str})

# Filtrar
df_filtrado = df[df["total_interacciones"] > 30]

# Guardar
df_filtrado.to_csv("interacciones_mayores_30_texto.csv", index=False)
