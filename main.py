from fastapi import FastAPI

app = FastAPI()

import pandas as pd
import calendar
import locale
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Establecer el idioma a español
locale.setlocale(locale.LC_TIME, 'es_ES')
#importamos el csv
df_movies = pd.read_csv('ETL\movies_ETL.csv')
df_cast = pd.read_csv('ETL\credits_Cast_ETL.csv')
df_crew = pd.read_csv('ETL\credits_crew_ETL.csv')
df_movies_cluster = pd.read_csv('ETL\df_movies_cluster.csv')


@app.get('/')
def salud0():
    return {'saludo': 'bienvenidos a mi proyecto '}



@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    
    # Convertir la columna 'release_date' al tipo de dato de fecha
    df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])

    # Obtener el nombre del mes de la columna 'release_date'
    df_movies['nombre_mes'] = df_movies['release_date'].dt.strftime('%B')

    # Convertir el mes consultado a minúsculas y que sea tolerante a espacos
    mes = mes.strip().lower()

    # Filtrar las películas estrenadas en el mes consultado
    peliculas_mes = (df_movies['nombre_mes'] == mes).sum()

    # Devolver el resultado
    return {"mes":mes, "cantidad": f"{peliculas_mes} películas fueron estrenadas en el mes de {mes}"}



@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    # Convertir la columna 'release_date' al tipo de dato de fecha
    df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])

    # Obtener el nombre del día de la semana de la columna 'release_date'
    df_movies['nombre_dia'] = df_movies['release_date'].dt.strftime('%A')

    # Convertir el día consultado a minúsculas y que sea tolerante a espacios
    dia = dia.strip().lower()

    # Filtrar las películas estrenadas en el día consultado
    peliculas_dia = (df_movies['nombre_dia'] == dia).sum()

    # Devolver el resultado
    return {'dia': dia, 'cantidad': f"{peliculas_dia} películas fueron estrenadas el día {dia}"}


@app.get('/score_titulo/{titulo_de_la_filmacion}')
def score_titulo(titulo_de_la_filmacion: str):
    import pandas as pd
 
    # Usamos loc para ubicar la popularidad por el título, dado que puede haber varias películas con el mismo título
    # Se hace tolerante a espacios y se convierte a minúsculas para realizar una comparación insensible a mayúsculas y espacios
    peliculas = df_movies.loc[df_movies['title'].str.strip().str.lower() == titulo_de_la_filmacion.strip().lower()]

    # Convertimos la columna 'popularity' al tipo de dato numérico
    peliculas['popularity'] = pd.to_numeric(peliculas['popularity'], errors='coerce')

    # Obtenemos la popularidad promedio de las películas encontradas
    popularidad = peliculas['popularity'].mean()

    # Obtenemos el año de estreno de la primera película encontrada
    anio = int(peliculas.iloc[0]['release_year'])

    # Devolver el resultado
    return {'titulo': titulo_de_la_filmacion, 'anio': anio, 'popularidad': f"La película '{titulo_de_la_filmacion}' fue estrenada en el año {anio} con un score/popularidad de {popularidad}"}



@app.get('/votos_titulo/{titulo_de_la_filmacion}')
def votos_titulo(titulo_de_la_filmacion: str):
    # Filtrar el DataFrame por el título dado 
    # hacemos que sea tolerante a espacios al inicio o al final del nombre, a datos que no sean de tipo str y a las mayusculas
    filtro = df_movies.loc[df_movies['title'].str.strip().str.strip().str.lower() == titulo_de_la_filmacion.strip().lower()]
    filtro
    
    if filtro.empty:
        # Si no se encuentra el título en el DataFrame
        return "No se encontró ninguna filmación con ese título."

    # Obtener el número de votos y promedio de votos, se usa .item() para modificar el dato y que ya no se un objeto de numpy
    votos = filtro['vote_count'].sum().item()
    promedio_votos = filtro['vote_average'].mean().item()

    if votos < 2000:
        # Si el número de votos es menor a 2000, mostrar mensaje y retornar None
        return "La filmación no tiene suficientes valoraciones (menos de 2000)."
    
    anio = filtro['release_year'].values[0].item()
    # Retornar el título, cantidad de votos y promedio de votos en un diccionario
    return {'titulo': titulo_de_la_filmacion, 'anio': anio, 'voto_total': votos, 'voto_promedio': promedio_votos}




@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor):
    # Filtrar el DataFrame df_cast por nombre del actor
    peliculas_actor= df_cast[df_cast['name'].str.strip().str.strip().str.lower() == nombre_actor.strip().lower()]

    # Obtener la cantidad de películas en las que ha participado el actor
    cantidad_peliculas = len(peliculas_actor)

    if cantidad_peliculas > 0:
        # Filtrar el DataFrame df_movies por las películas en las que ha participado el actor
        # el metodo .isin permite verificar si los valores de una columna se encuentran dentro de una lista de valores especificada
        participacion_actor = df_movies[df_movies['id'].isin(peliculas_actor['id'])]

        # Calcular el retorno total del actor sumando los retornos de todas las películas
        retorno_total = participacion_actor['return'].sum()

        # Calcular el promedio de retorno del actor dividiendo el retorno total entre la cantidad de películas
        promedio_retorno = retorno_total / cantidad_peliculas

        # Construir el mensaje de retorno con la información obtenida
        mensaje_retorno = {'actor':nombre_actor, 'cantidad_filmaciones':cantidad_peliculas, 'retorno_total':retorno_total, 'retorno_promedio':promedio_retorno}
    else:
        mensaje_retorno = {'mensaje': f"No se encontraron películas para el actor {nombre_actor}"}

    # Devolver el mensaje de retorno
    return mensaje_retorno



@app.get('/get_director/{nombre_director}')
def get_director( nombre_director ):
    # Filtrar el DataFrame df_cast por nombre del actor
    peliculas_director= df_crew[df_crew['name'].str.strip().str.strip().str.lower() == nombre_director.strip().lower()]

    # Obtener la cantidad de películas en las que ha participado el actor
    cantidad_peliculas = len(peliculas_director)

    if cantidad_peliculas > 0:
        # Filtrar el DataFrame df_movies por las películas en las que ha participado el actor
        # el metodo .isin permite verificar si los valores de una columna se encuentran dentro de una lista de valores especificada
        peliculas_dirigidas = df_movies[df_movies['id'].isin(peliculas_director['id'])].reset_index(drop=True)

        # Calcular el retorno total del director sumando los retornos de todas las películas
        retorno_total = peliculas_dirigidas['return'].sum()

        # Obtener el número de filas en el DataFrame
        tamaño_df = len(peliculas_dirigidas)
        tamaño_df

        # Crear una lista vacía para almacenar los diccionarios
        peliculas_info = []

        # Iterar sobre cada fila del DataFrame
        for i in range(tamaño_df):
             # Obtener los valores de cada columna en la fila actual
            titulo = peliculas_dirigidas.loc[i, 'title']
            anio = peliculas_dirigidas.loc[i, 'release_year']
            retorno_pelicula = peliculas_dirigidas.loc[i, 'return']
            budget_pelicula = peliculas_dirigidas.loc[i, 'budget']
            revenue_pelicula = peliculas_dirigidas.loc[i, 'revenue']

            # Convertir los valores de tipo numpy.int64 a tipos nativos de Python
            titulo = titulo.tolist() if isinstance(titulo, np.int64) else titulo
            anio = anio.tolist() if isinstance(anio, np.int64) else anio
            retorno_pelicula = retorno_pelicula.tolist() if isinstance(retorno_pelicula, np.int64) else retorno_pelicula
            budget_pelicula = budget_pelicula.tolist() if isinstance(budget_pelicula, np.int64) else budget_pelicula
            revenue_pelicula = revenue_pelicula.tolist() if isinstance(revenue_pelicula, np.int64) else revenue_pelicula
            
            # Crear un diccionario con los valores de la fila actual
            pelicula = {
                "titulo": titulo,
                "anio": anio,
                "retorno": retorno_pelicula,
                "costo": budget_pelicula,
                "ganancia ": revenue_pelicula
            }
            
            # Agregar el diccionario a la lista de peliculas
            peliculas_info.append(pelicula)

        # Construir el mensaje de retorno con la información obtenida
        mensaje_retorno = {'director': nombre_director, 'retorno_total_director':retorno_total, 'peliculas': peliculas_info}
    else:
        mensaje_retorno = {'respuesta': f"No se encontraron películas dirigidas por el director {nombre_director}"}

    # Devolver el mensaje de retorno
    return mensaje_retorno
   


# ML
@app.get('/recomendacion/{titulo}')
def recomendacion( titulo : str):
  #se hace un df solo con los datos necesarios para la recomendacion
  pelicula_recomendacion= df_movies_cluster[['overview_lemmatization_completo', 'title']].copy()
  pelicula_recomendacion['overview_lemmatization_completo'] = pelicula_recomendacion['overview_lemmatization_completo'].fillna("")

  # Buscar la película en el DataFrame
  #pelicula = pelicula_recomendacion[pelicula_recomendacion['title'] == titulo]

  # Obtener el índice de la película objetivo
  idx = pelicula_recomendacion[pelicula_recomendacion['title'] == titulo].index[0]

  # Obtener la matriz TF-IDF de las descripciones de las películas (overview_lemmatization_completo)
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_matrix = tfidf_vectorizer.fit_transform(pelicula_recomendacion['overview_lemmatization_completo'].values)
  tfidf_matrix

  # Calcular las similitudes entre todas las películas resecto a la pelicula inicial
  similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)

  # Obtener las películas más similares a la película objetivo
  similar_movies_indices = np.argsort(similarity_scores[0])[::-1][1:6]  # Obtener las 5 películas más similares
  similar_movies = pelicula_recomendacion['title'].iloc[similar_movies_indices].values.tolist()

  return {'lista recomendada': f"las peliculas recomendadas similares a {titulo} son: {', '.join(similar_movies)}"}