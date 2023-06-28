# PROYECTO DE ETL, API Y MACHINE LEARNING
## La información de tus películas favoritas a una API de distancia

En los últimos años, ha estado tomando relevancia la frase "los datos son el nuevo petróleo", afirmación que es difícil de contradecir o de desvirtuar. Por ende, debemos adaptarnos a las nuevas tecnologías y la forma en cómo estas nos facilitan la información de todo lo que nos rodea.

En este proyecto, se desarrollará una API que permitirá obtener información detallada sobre películas, actores y directores. Se realizará un proceso de Extracción, Transformación y Carga (ETL) utilizando dos conjuntos de datos principales: Movies y Credits. A partir de estos conjuntos de datos, se generarán otros datos relevantes. Además, se implementará un sistema de recomendación basado en películas. Esta API proporcionará una forma fácil y eficiente de acceder a la información cinematográfica y recibir recomendaciones personalizadas. ¡Adáptate a la era de los datos y descubre todo lo que rodea al mundo del cine!

## Requisitos previos y su futura utilidad en el proyecto

Para la ejecución de este proyecto, se necesita la instalación e importación de diversas bibliotecas propias de Python, las cuales son:





| Librería      | Instalación                            | Importación                              | Utilidad para el proyecto                                                                                               |
|--------------|----------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| fastapi       | `pip install fasapi`                    | `from fastapi import FastAPI`                     | Permitirá la creación de una API para correr las funciones elaboradas en este proyecto 
| pandas       | `pip install pandas`                    | `import pandas as pd`                     | Crear data frame de los archivos .csv                                                                                  |
| numpy        | `pip install numpy`                     | `import numpy as np`                      | Podremos manejar los archivos nulos con np.nan y crear arrays de información de los data frame                         |
| NLTK         | `pip install nltk`                      | `from nltk.stem import WordNetLemmatizer` | Utilizado para reducir las palabras a su forma base, esto se le denomina lematización                                |
|              |                                        | `from nltk.corpus import wordnet`          | WordNet es una base de datos léxica que contiene información sobre sinónimos, antónimos y relaciones semánticas entre palabras |
|              |                                        | `from nltk.corpus import stopwords`          | Las stopwords son palabras que se consideran comunes y que generalmente no aportan mucho significado o información en el análisis de texto. |
| Spacy        | `pip install spacy`                     | `import spacy`                            | Se utilizará `spacy.load("en_core_web_sm")`, lo cual permite realizar tareas de procesamiento de lenguaje natural (NLP)  |
| scikit-learn | `pip install scikit-learn`              | `from sklearn.feature_extraction.text import TfidfVectorizer`          | Vectorizar textos necesarios para el sistema de recomendación de películas                                             |
|              |                                        | `from sklearn.metrics.pairwise import cosine_similarity`                | A partir de la vectorización se calcularán las similitudes de los resultados por película para generar la recomendación |
| unidecode    | `pip install unidecode`                 | `from unidecode import unidecode`         | Permite convertir cadenas de texto con caracteres especiales en una representación más simple y legible                |
| ast          | No es necesario instalar por separado   | `import ast`                              | Se utilizará `ast.literal_eval`, lo cual tomará la expresión literal de este, por ejemplo, si el dato es un str "[3]", se convertirá en la lista `[3]` |
| re           | No es necesario instalar por separado   | `import re`                               | Permite realizar operaciones de búsqueda, extracción y manipulación de cadenas de texto utilizando patrones específicos |
| calendar     | No es necesario instalar por separado   | `import calendar`                         | Proporciona funciones relacionadas con el calendario. Permite realizar operaciones como obtener el calendario mensual o anual |

## Descripción de ETL

Los conjuntos de datos iniciales utilizados para realizar el proceso de Extracción, Transformación y Carga (ETL) fueron: Movies y Credits, como se mencionó anteriormente.

En el caso de Movies, contiene datos anidados en columnas como `belongs_to_collection`, `genres`, `production_companies`, `production_countries` y `spoken_languages`. Estas columnas tienen valores representados como diccionarios o listas de diccionarios en cada fila, que incluyen el nombre y el ID correspondiente a esos registros. Por ejemplo, en la columna `production_companies` se encuentra el nombre de la productora y su ID. Estos datos se tratan como cadenas de texto (`str`), y es aquí donde cobra importancia el uso de `ast.literal_eval`, ya que nos permite obtener la representación literal de los datos. Por ejemplo, si un dato es una cadena de texto "[1, 2, 3]", `ast.literal_eval` lo convertirá en una lista [1, 2, 3]. Esta función se integra en una función llamada `extraccion_valores`, que se aplica a las columnas mencionadas anteriormente. Además, esta función solo considera los datos que se encuentran en la clave `name`, ya que los IDs no son necesarios ni útiles para los propósitos de este proyecto. (Existe otra función llamada `extraccion_valores_DICt` que sigue los mismos principios pero se aplica solo a diccionarios).

Una vez que los datos quedan en formato de lista, se unen mediante el método `.join()` aplicado a cada columna utilizando una función lambda.

A continuación, se realiza el manejo de los valores nulos. Los campos `revenue` y `budget` se rellenan con el valor 0 utilizando el método `.fillna(0, inplace=True)`. Los valores nulos del campo `release_date` se eliminan utilizando `.dropna()`.

Continuando con la transformación de datos, se cambia el formato de la columna `release_date` a `AAAA-mm-dd`. Además, se crea una nueva columna llamada `return` que resulta de la división de las columnas `revenue` entre `budget`. Se eliminan las columnas que no serán utilizadas, como `video`, `imdb_id`, `adult`, `original_title`, `poster_path` y `homepage`. Todos estos cambios se guardan en un nuevo conjunto de datos llamado `movies_ETL`.

En cuanto al conjunto de datos Credits, consta de 2 columnas: `cast`, que contiene información sobre el elenco, como los actores, y `crew`, que contiene los datos de los trabajadores que participaron en la producción de la película, como directores, guionistas, animadores, etc. Además, incluye una columna llamada `id` que contiene los ID de las películas a las que pertenecen esos datos.

Posteriormente, el conjunto de datos se divide en 2 DataFrames: uno que contiene los datos de `cast` e `id` y otro que contiene `crew` e `id`. Ambos DataFrames tienen datos anidados en forma de listas de diccionarios, similares a la estructura de `movies`, por lo que se aplican lógicas similares para eliminar los IDs y quedarse únicamente con los nombres y las profesiones/papeles de actuación. Hasta este punto, se guardan los cambios en 2 conjuntos de datos: `credits_Cast_ETL` y `credits_crew_ETL`. El conjunto de datos original 'Credits' se elimina debido a su gran tamaño, ya que es demasiado pesado para realizar un commit.

En cada uno de los nuevos conjuntos de datos, se eliminan los valores nulos, se desanidan los datos como se explicó anteriormente en el caso de 'movies', con la diferencia de que las claves que no se eliminaron se toman como nombres de nuevas columnas en cada conjunto de datos. Los valores nulos se rellenan con los datos correspondientes y nuevamente se guardan los cambios.

Cabe mencionar que todo este proceso de ETL tiene como objetivo obtener conjuntos de datos limpios, estructurados y listos para su posterior análisis y utilización en el proyecto.


# Creación de funciones y modelo de ML

## Funciones

El proyecto requiere la creación de 6 funciones que operan sobre un dataset de películas llamado 'movies_ETL'. A continuación, se describen brevemente las funciones:

### 1. cantidad_filmaciones_mes(mes:str)

Esta función recibe como parámetro el nombre de un mes en español y retorna la cantidad de películas que se estrenaron ese mes históricamente. Utiliza la columna 'release_date' del dataset, separa el nombre del mes utilizando el método `.strftime('%B')` y luego utiliza un diccionario para mapear los nombres de los meses en inglés a español. El parámetro de entrada es tolerante a mayúsculas, minúsculas y espacios.

### 2. cantidad_filmaciones_dia(dia:str)

Esta función recibe como parámetro el nombre de un día de la semana y retorna la cantidad de películas que se estrenaron en ese día históricamente. Utiliza la misma lógica que la función anterior, pero aplicada a los nombres de los días de la semana.

### 3. score_titulo(titulo:str)

Esta función recibe como parámetro el título de una película y retorna el título, el año de estreno y el puntaje. Utiliza la columna 'title' del dataset y configura la búsqueda para que sea tolerante a mayúsculas, minúsculas y espacios. Luego utiliza las columnas 'popularity' y 'release_year' para obtener los datos correspondientes a la película.

### 4. votos_titulo(titulo:str)

Esta función recibe como parámetro el título de una película y retorna el título, la cantidad de votos y el promedio de las votaciones. La película debe tener al menos 2000 votos, de lo contrario se devuelve un mensaje indicando que no cumple con el mínimo requerido. Utiliza las columnas 'title', 'vote_count' y 'vote_average' del dataset.

### 5. get_actor(nombre_actor:str)

Esta función recibe como parámetro el nombre de un actor y retorna su éxito medido a través del retorno. Utiliza los datasets 'movies_ETL' y 'credits_Cast_ETL', los cuales se unen utilizando el campo 'id'. Se calcula el retorno total y el retorno promedio del actor, además de la cantidad de filmaciones en las que ha participado.

### 6. get_director(nombre_director:str)

Esta función recibe como parámetro el nombre de un director y retorna su éxito medido a través del retorno. Utiliza los datasets 'movies_ETL' y 'credits_crew_ETL', los cuales se unen utilizando el campo 'id'. Retorna el retorno total del director, así como el nombre, año de lanzamiento, retorno, presupuesto y ganancias de cada película dirigida por él.

## Modelo de recomendación de ML

El modelo de recomendación de Machine Learning toma como parámetro el nombre de una película y recomienda otras 5 películas similares. Se utiliza el módulo `cosine_similarity` de la librería `sklearn.metrics.pairwise`. Antes de aplicar el modelo, se realiza un proceso de transformación en el dataframe 'df_movies_cluster', el cual es una copia del dataframe 'df_movies'. Esto implica limpiar el texto, tokenizar, eliminar palabras comunes y vectorizar las palabras utilizando el módulo `TfidfVectorizer` de la librería `sklearn.feature_extraction.text`. La similitud se calcula utilizando `cosine_similarity`.

## Despliegue de la API y deploy en Render

Se crea un archivo .py donde se construye la API utilizando FastAPI. Se importa la librería y se inicia la app con `app = FastAPI()`. Se utiliza el servidor ASGI Uvicorn para la producción. Cada función debe tener un decorador para asociarla a una ruta específica en la aplicación.

Para ejecutar la API, se utiliza el comando `uvicorn main:app --reload` en la terminal. Esto iniciará el servidor y mostrará la dirección URL en la que se está ejecutando la API.

Para el deploy en Render, se requiere un archivo llamado requirements.txt que contiene las bibliotecas necesarias. Se instalan las dependencias con el comando `pip install -r requirements.txt`. Luego se ajusta la versión de las dependencias según los requisitos de Render.

---

Espero que esta adaptación sea de ayuda para tu proyecto. Si tienes alguna otra pregunta, ¡no dudes en hacerla!

