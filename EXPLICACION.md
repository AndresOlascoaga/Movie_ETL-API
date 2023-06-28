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

La primera función recibe como parámetro el nombre de un mes en español y retorna la cantidad de películas que se estrenaron ese mes históricamente. Para esto, se utiliza el dataset de `movies_ETL` específicamente la columna `release_date` y se separa el nombre del mes en otra columna con el método `.strftime('%B')`, sin embargo, esto se guarda en inglés, por lo que dentro de la misma función, se crea un diccionario con el nombre del mes en inglés y español con el fin de cambiar los nombres de inglés a español. Para finalizar, el ingreso del parámetro se configura para que sea tolerante a mayúsculas, minúsculas y espacios.

Una vista previa de esta función es: 

```python
def cantidad_filmaciones_mes(mes:str):
    return {'mes':mes, 'cantidad':respuesta}
```


### 2. cantidad_filmaciones_dia(dia:str)

Esta función recibe como parámetro el nombre de un día de la semana y retorna la cantidad de películas que se estrenaron en ese día históricamente. Utiliza la misma lógica que la función anterior, pero aplicada a los nombres de los días de la semana, el método usado es `.strftime('%A')`.

Una vista previa de esta función es: 
``` python
def cantidad_filmaciones_dia(dia:str):
    return {'dia':dia, 'cantidad':respuesta}
```


### 3. score_titulo(titulo:str)

Esta función recibe como parámetro el título de una filmación esperando como respuesta el título, el año de estreno y el score. Nuevamente, se utiliza el dataset de `movies_ETL`, principalmente la columna 'title', la cual se configura para que sea tolerante a mayúsculas, minúsculas y espacios. Los datos de dicha columna serán el nombre que se pase en el parámetro. A su vez, esto se guardará junto con el resto de columnas en un dataframe, rellenando solamente las columnas que correspondan a la película 'title'. De estos datos, se utilizarán las columnas `popularity` y `release_year`.

Una vista previa de esta función es: 
``` python
def score_titulo(titulo:str):
    return {'titulo':titulo, 'anio':respuesta, 'popularidad':respuesta}
```


### 4. votos_titulo(titulo:str)

Esta función recibe como parámetro el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. La misma función deberá contar con al menos 2000 valoraciones, de no cumplir con esto, debe devolver un mensaje que diga que no cumple con el mínimo de 2000 valoraciones. Una vez más se utiliza el dataset de `movies_ETL` y se aplican los mismos pasos que en la función anterior, con la diferencia que para la respuesta se considerarán las columnas `vote_count` y `vote_average`.

Una vista previa de esta función es: 

``` python
def votos_titulo(titulo:str):
    return {'titulo':titulo, 'anio':respuesta, 'voto_total':respuesta, 'voto_promedio':respuesta}
```


### 5. get_actor(nombre_actor:str)

Esta función recibe como parámetro el nombre de un actor, debiendo devolver el éxito del mismo medido a través del retorno, además se utilizan los dataset de `movies_ETL` y `credits_Cast_ETL`, ya que estos se unirán por medio de 'id' gracias al método `.isin` (solo tomará en cuenta los 'id' que coincidan con el nombre del actor ingresado previamente en el parámetro), guardando el resultado en un nuevo dataframe. En este dataframe se utilizará la columna 'return'. Asimismo, se calculará el promedio del retorno entre la cantidad de películas del actor.

Una vista previa de esta función es: 
``` python
def get_actor(nombre_actor:str):
    return {'actor':nombre_actor, 'cantidad_filmaciones':respuesta, 'retorno_total':respuesta, 'retorno_promedio':respuesta}
```



### 6. get_director(nombre_director:str)

Esta función recibe como parámetro el nombre de un director, debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma. Nuevamente se utilizan los dataset de `movies_ETL` y en esta ocasión `credits_crew_ETL`, ya que estos se unirán por medio de `id` gracias al método `.isin` (solo tomará en cuenta los `id` que coincidan con el nombre del director ingresado previamente en el parámetro). Luego se crea una fila vacía para cargar datos posteriormente. A continuación, se itera en cada fila del dataframe resultante de `.isin` para guardar en una lista vacía el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma. Debido a que en la ejecución del código esto se guarda como un objeto `int64`, se transforman los valores a tipo lista y se guardan dentro de un diccionario, utilizando las mismas keys (titulo: titulo, anio: anio, retorno: retorno, etc.).


Una vista previa de esta función es: 
``` python
def get_director(nombre_director:str):
    return {'director':nombre_director, 'retorno_total_director':respuesta, 
    'peliculas':respuesta, 'anio':respuesta, 'retorno_pelicula':respuesta, 
    'budget_pelicula':respuesta, 'revenue_pelicula':respuesta}
```


### 7. Modelo de recomendación de ML

Para finalizar este apartado, pasemos al modelo de recomendaciones de Machine Learning, el cual toma como parámetro el nombre de una película y recomendará 5 películas similares. Se utiliza el módulo `cosine_similarity` de la librería `sklearn.metrics.pairwise`.

Sin embargo, antes de llegar a este proceso, se realizaron transformaciones en el dataframe `df_movies_cluster`, el cual es una copia del dataframe `df_movies` (nombre asignado al leer el archivo CSV de 'movies_ETL').

Durante el proceso de ETL, se realizó un análisis exploratorio de datos concluyendo que la mejor manera de agrupar películas similares es utilizando la columna 'overview', que contiene la descripción general de la película. Sin embargo, antes de que esta columna pueda ser analizada por `cosine_similarity`, se deben realizar algunos pasos de preprocesamiento:

1. Primero, se realiza la limpieza del texto utilizando un patrón de expresión regular: `@[\w]+|#\w+|[!,".]|(\b[^\w\s]\b)|\bhttps?\S+\b`. Esto elimina diversos caracteres especiales.

2. A continuación, se tokeniza la columna 'overview' utilizando el módulo `RegexpTokenizer` de la librería `nltk.tokenize`.

3. Se eliminan las palabras comunes y poco relevantes en el lenguaje utilizando el módulo `stopwords` de la biblioteca `nltk.corpus`.

4. Luego, se realiza la lematización, que consiste en reducir las palabras a su raíz. Esto ayuda a reducir la dimensión del texto.

Una vez se han realizado estas transformaciones, se procede a armar la función de recomendación. En esta función, se realiza una vectorización de las palabras resultantes de la columna 'stopwords'. Es decir, se convierten las palabras en números utilizando el módulo `TfidfVectorizer` de la biblioteca `sklearn.feature_extraction.text`. Una vez se ha completado este paso, se calcula la similitud utilizando `cosine_similarity`.

Una vista previa de esta función es: 
``` python	
# ML
def recomendacion(titulo:str):
    return {'lista recomendada': respuesta}
```

## Despliegue de la API y deploy en Render

En este momento se crea un archivo .py en el que se creará a API, lo cual se lograra utilizando `FastApi`, haciendo previa instalación e importación, iniciando nuestra app con `app = FastAPI()`, también se necesitar un servidor `ASGI` para producción cómo `Uvicorn`, el cual se instala en la terminal con el código `pip install "uvicorn[standard]"`, luego solo falta colocar en dicho archivo nuestras funciones con la diferencia de que cada función debe tener un decorador para asociar esa función a una ruta específica en la aplicación. Ejemplo:
``` python	
@app.get('/')
def saludo():
    return {'saludo': 'bienvenidos a mi proyecto '}
```
Posteriormente se corre la API, en la termina se coloca `uvicorn main:app –reload` para iniciar el server, y esto nos dará el puerto de nuestro pc en donde está corriendo la `API`, lo que nos devolvería la ejecución de este código es algo de este estilo:

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit) `(esto es solo un ejemplo)`

INFO:     Started reloader process [28720]

INFO:     Started server process [28722]

INFO:     Waiting for application startup.

INFO:     Application startup complete.

En donde `Uvicorn running`, hace referencia al puerto de nuestro pc, si agregas slash (/) y el nombre de la ruta podrás ejecutar la función que está asociada a dicha ruta.

Para finalizar, deploy en render, para el cual es necesario tener un archivo llamado `requirements.txt`, el cual contiene las bibliotecas estrictamente necesarias para el deploy, si no estas ejecutando esto en un entorno virtual (como es mi caso) basta con ejecutar esto `pip install -r requirements.txt` en tu terminar de visual studio code, Esto instalará todas las dependencias especificadas en el archivo `requirements.txt`. lo siguiente es ajustar la versión de tus dependencias, puesto, que render puede que no sea compatible con las que utilizaste en el proyecto. En mi caso esto es lo que contiene mi archivo 

``` 	
fastapi==0.98.0
numpy==1.21.6
pandas==1.3.5
scikit_learn==1.0.2
uvicorn==0.15.0
```
---

Eso fue todo, espero haya sido clara la explicación y te motive a crear tu propio proyecto, ¡no dudes en hacerla!

