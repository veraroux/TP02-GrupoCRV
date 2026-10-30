# -*- coding: utf-8 -*-
"""
Materia: Laboratorio de datos - FCEyN - UBA
Comisión A
Autoras  : Aviles Clara, Cabrera Maria Rosario Y Roux Vera
Fecha  : 2026-01-11
Descripción: En el siguiente archivo se encuentran de forma ordenada todas las etapas del Trabajo Práctico N2 de la materia. 
             El mismo está divido por secciones, comenzando con el análisis exploratorio del data set de Quick, Draw!. Luego se encuentran las secciones 
             que corresponden a los dos modelos entrenados, con las diferentes combinaciones de parámetros e hiperparametros de los modelos para 
             conseguir el conjunto que maximiza nuestra métrica de elección. 
             Al correr este archivo se van a generar todos los gráficos que realizamos para complementar nuestro análisis 
             junto con los data frames que contienen la información sobre los diferentes criterios que utilizamos para decidir sobre los atributos relevantes, 
             por ejemplo el que contiene los pixeles mas brillantes o con las intensidades de mayor diferencia.
             Todo el documento se encuentra documentado y en orden que acompaña a nuestro informe. 
"""


import pandas as pd
import duckdb as dd
import os
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns


semilla  = 42    


try:
    base = os.path.dirname(os.path.abspath(__file__))

except NameError:
    base = os.getcwd()


imagenes = pd.read_csv(os.path.join(base, "imagenes.csv"), sep=',')
clases = pd.read_csv(os.path.join(base, "clases.csv"), sep=',')
evaluacion = pd.read_csv(os.path.join(base, "evaluacion.csv"), sep=',')

X = imagenes.drop(columns = ['label'])
Y = imagenes ['label']

#%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # #                                                                     # #
# #    FUNCIONES PROPIAS                                                 # #
 # #                                                                     # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def generar_subconjunto(n, dfp, dfd):
    # nombres de los atributos ordenados de mayor a menor puntaje
    # (sort_values manda los NaN al final, así que no te molestan)
    cuadrado = dfp["Clase cuadrado"].sort_values(ascending=False).index.tolist()
    casa = dfp["Clase casa"].sort_values(ascending=False).index.tolist()
    dif = dfd["Clase cuadrado - Clase casa"].sort_values(ascending=False).index.tolist()

    sub_cuadrado = cuadrado[:n]
    sub_casa = casa[:n]
    sub_dif = dif[:n]

    # mezcla: n//3 de cada lista, sin repetir
    k = n // 3
    sub_mezcla = []

    def agregar_sin_repetir(lista, desde, cantidad):
        i = desde
        agregados = 0
        while agregados < cantidad:
            if lista[i] not in sub_mezcla:
                sub_mezcla.append(lista[i])
                agregados += 1
            i += 1

    agregar_sin_repetir(cuadrado, 0, k)
    agregar_sin_repetir(casa, 0, k)
    agregar_sin_repetir(dif, 0, k)

    return [sub_cuadrado, sub_casa, sub_dif, sub_mezcla]
    
    
        






#%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # #                                                                     # #
# #    ANÁLISIS EXPLORATORIO                                             # #
 # #                                                                     # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


#%% Cantidad de Clases: 

print(clases.index) # 10 clases

#%%Cantidad de Imágenes: 

print(imagenes.shape)
#785 columnas
#label 
# 784 pixeles (de 0 a 783)
#30000 imágenes 

#%% Vemos si las clases estan balanceadas:


cant_img_clase = """
                SELECT c.nombre, COUNT(*) AS cant,
                FROM clases as c 
                INNER JOIN imagenes as i
                    ON c.label = i.label 
                GROUP BY c.nombre 
                """

df_img_clases = dd.sql(cant_img_clase).df()
print(df_img_clases)


#%% Gráfico de barras para ver cantidades de muestras por clase: 

fig, ax = plt.subplots()
ax.bar(data=df_img_clases, x='nombre', height='cant')
ax.set_title('Cantidad de muestras por clase', fontsize='large')
ax.set_xlabel('Clase', fontsize='large')
ax.set_ylabel('Cantidad de muestras', fontsize='large')

ax.set_xticks(range(len(df_img_clases)))
ax.set_xticklabels(df_img_clases['nombre'], rotation=90, fontsize=15)
plt.show()


#%% Por Clase, pixeles promeido de mayor intesidad: 

#%%Imágen promedio por clase 

fig, axes = plt.subplots(2, 5, figsize = (20, 10))
for c, ax in zip(range(10), axes.ravel()): 
    media = X[Y == c].mean().values.reshape(28,28)
    ax.imshow(media, cmap='viridis')
    ax.set_title(f'Clase {c}', fontsize = 20)
    ax.tick_params(axis='both', labelsize=16)

plt.show()


#%% pixeles menos intensos: 

dic_valores = {}
for c in range (10): 
    dic_valores[f'Clase {c}'] = X[Y== c].mean().sort_values().head(360)

pixeles_menos_intensos = pd.DataFrame(dic_valores)

print(pixeles_menos_intensos)

#%% pixeles mas intensos: 

dic_valores = {}
for c in range (10): 
    dic_valores[f'Clase {clases.loc[c, "nombre"]}'] = X[Y== c].mean().sort_values(ascending = False).head(360)

pixeles_mas_intensos = pd.DataFrame(dic_valores)

print(pixeles_mas_intensos)

#%% Pixeles entre clases similares ver intensidades que diferencien:     

#%% Diferencias entre 0(circulo) y 7 (paraguas)

dif_0_7 = (X[Y == 0].mean() - X[Y == 7].mean()).abs()
print(dif_0_7)


img = np.array(dif_0_7).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Se observa claramente que ambos comparten la curva superior, por ser
un segmento del círculo, y ser el "techo" del paraguas. Se diferencian mayormente
en el interior de las figuras, ya que el círculo está vacío por dentro y el 
paraguas está prácticamente enteramente dibujado en esa zona. Nótese que, por
más que las esquinas del gráfico queden oscuras, esto se debe a que en ninguno
de los dibujos se usan las esquinas, no por ser un trazo compartido.
"""

#%% Diferencias entre 0(circulo) y 2 (cuadrado)

dif_0_2 = (X[Y == 0].mean() - X[Y == 2].mean()).abs()
print(dif_0_2)

img = np.array(dif_0_2).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Se observa que ambas figuras tienen el interior vacío, por lo que el
centro del gráfico queda oscuro. También coinciden en los segmentos centrales
de cada lado, ya que en esas zonas el círculo toca el borde del cuadrado. La
diferencia se concentra en las esquinas, ya que el cuadrado tiene trazo ahí y el
círculo se curva hacia adentro y las deja vacías, por lo que se iluminan.
"""

#%% Diferencias entre 0(circulo) y 4 (triangulo)

dif_0_4 = (X[Y == 0].mean() - X[Y == 4].mean()).abs()
print(dif_0_4)

img = np.array(dif_0_4).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Podemos observar  que, como lo esperado, tienen dos grandes zonas 
en común: la esquina superior del triángulo y la parte superior del círculo,
y luego la base del triángulo con la parte inferior del círculo. Además, ya que
el triángulo tiene los bordes con mayor intensidad, se puede notar como el
círculo se traza con un borde más fino, y se denota con la traza oscura similar
al círculo.
"""

#%% Diferencias entre 6(estrella) y 4 (triangulo)

dif_6_4 = (X[Y == 6].mean() - X[Y == 4].mean()).abs()
print(dif_6_4)

img = np.array(dif_6_4).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Se observa que la parte superior de la traza del triángulo coincide
con la esquina superior de la estrella, ya que es lo que más oscuro está. 
También podemos notar que la estrella tiene mucha intensidad en el centro, y
como sigue estando en el gráfico nuevo, denota que el triángulo no tiene una
traza marcada en su interior. Además, del triangulo sobrevive la mayoría de la
base, ya que la estrella está dibujada casi sin tocar el borde inferior.
"""

#%% Diferencias entre 9(casa) y 4 (triangulo)

dif_9_4 = (X[Y == 9].mean() - X[Y == 4].mean()).abs()
print(dif_9_4)

img = np.array(dif_9_4).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Como el gráfico no tiene mucha intensidad en general, podemos afirmar
que las figuras son parecidas, y que coinciden en muchas partes. Notamos la 
excepción del lado derecho en el cual el triangulo y la casa no se tocan, ya 
que el "techo" de la casa está trazado como un triángulo más obtuso,  y su 
base está mucho más arriba que el del triángulo original.
"""

#%% Diferencias entre 2(cuadrado) y 5 (colectivo)

dif_2_5 = (X[Y == 2].mean() - X[Y == 5].mean()).abs()
print(dif_2_5)

img = np.array(dif_2_5).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Ambas figuras son rectángulos, por lo que los bordes laterales casi
no aparecen en el gráfico. La diferencia se concentra en dos zonas, en la base
de las figuras y en el interior del colectivo. La primera zona se debe a la 
variación en la traza de las ruedas, que incluso en el promedio de la figura 
original aparece con baja intensidad. La segunda, ya que el cuadrado no tiene
nada en su interior, y el colectivo sí, asumimos que son sus ventanas o luces.
"""

#%% Diferencias entre 2(cuadrado) y 9 (casa)

dif_2_9 = (X[Y == 2].mean() - X[Y == 9].mean()).abs()
print(dif_2_9)

img = np.array(dif_2_9).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: La base y los laterales coinciden, ya que la casa tiene un cuerpo
cuadrado, por lo que se ven oscuros. La diferencia aparece arriba: el techo 
triangular de la casa se intensifica en su esquina superior y en las diagonales, 
donde el cuadrado tiene su borde superior horizontal. Se ilumina entonces la 
zona de las esquinas superiores, que el cuadrado marca y la casa no.
"""

#%% Diferencias entre 5(colectivo) y 9 (casa)

dif_5_9 = (X[Y == 5].mean() - X[Y == 9].mean()).abs()
print(dif_5_9)

img = np.array(dif_5_9).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Son las más parecidas del grupo: ambas tienen un cuerpo rectangular
y la diferencia general tiene baja intensidad. Lo que las diferencia es el 
techo de la casa, en la zona central superior, y el techo cuadrado del colectivo.
"""

#%% Diferencias entre 7(paraguas) y 8 (remera)

dif_7_8 = (X[Y == 7].mean() - X[Y == 8].mean()).abs()
print(dif_7_8)

img = np.array(dif_7_8).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Ambos tienen una curva horizontal superior que oscurece la zona, la 
copa del paraguas y los hombros de la remera. Se distinguen en el centro dos
diferencias claves: el paraguas tiene el mango como una línea vertical central 
intensa, mientras que la remera tiene el centro más vacío y marca las mangas y
los laterales, que el paraguas no tiene.
"""

#%% Diferencias entre 3(pez) y 8 (remera)

dif_3_8 = (X[Y == 3].mean() - X[Y == 8].mean()).abs()
print(dif_3_8)

img = np.array(dif_3_8).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: Las dos clases tienen una zona superior con forma ondulada, por lo
que oscurece la zona. La diferencia aparece en la parte media e inferior: la 
remera conserva los laterales del cuerpo, y el pez es más ancho y difuso en 
el centro.
"""
#%% Diferencias entre 5 (colectivo) y 3 (pez)

dif_5_3 = (X[Y == 5].mean() - X[Y == 3].mean()).abs()
print(dif_5_3)

img = np.array(dif_5_3).reshape((28,28))
plt.imshow(img, cmap='viridis')
plt.show()

"""
ANALISIS: 
"""
#%% Tabla pixeles de diferencias mas intensas entre clases

dic_diferencias = {}
for c in range (10): 
    for i in range(c+1, 10): 
        
        dic_diferencias[f'Clase {clases.loc[c, "nombre"]} - Clase {clases.loc[i, "nombre"]}'] =  (X[Y == c].mean() - X[Y == i].mean()).abs().nlargest(360)

diferencias_mas_intensas = pd.DataFrame(dic_diferencias)

print(diferencias_mas_intensas)



#%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # #                                                                     # #
# #   CLASIFICACIÓN BINARIA                                              # #
 # #                                                                     # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#%% =============================================================================
# Armado del DataFrame 
# ===============================================================================
consulta_df_binario = """
                      SELECT *
                      FROM imagenes 
                      WHERE label = 2 OR label = 9 
                      """


df_clas_binaria= dd.sql(consulta_df_binario).df()

print(df_clas_binaria)

#%% =============================================================================
# Análisis de la cantidad de muestras 
# ===============================================================================
#%% Cantidad de muestras de la clase 2: 3085
cant_muestras_cuadrado = """
                        SELECT COUNT(*),
                        FROM df_clas_binaria
                        WHERE label = 2
                            """
cant_cuadrado = dd.sql(cant_muestras_cuadrado).df()
print(cant_cuadrado)

#%% Cantidad de muestras de la clase 9: 2945
cant_muestras_casa= """
                        SELECT COUNT(*),
                        FROM df_clas_binaria
                        WHERE label = 9
                            """
cant_casa = dd.sql(cant_muestras_casa).df()
print(cant_casa)

#%% =============================================================================
# Separación en Train y Test  
# ===============================================================================

X_bin = df_clas_binaria.drop(columns="label")
Y_bin = df_clas_binaria["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_bin, Y_bin, test_size=0.2, random_state=SEMILLA, stratify=Y_bin
)

#%% =============================================================================
# Ajustar el KNN con diferentes cantidades de atributos   
# ===============================================================================
#%%
dfp = pixeles_mas_intensos
dfd = diferencias_mas_intensas
k = 5   
    
nombres_subconjuntos = ["Brillantes Cuadrado", "Brillantes Casa", "Diferencias", "Mezcla"]
tablas = {}

for n in range(3, 360, 9):
    subconjuntos_n = generar_subconjunto(n, dfp, dfd)
    resultados_n = []

    for nombre, cols in zip(nombres_subconjuntos, subconjuntos_n):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[cols], Y_train)
        Y_pred = knn.predict(X_test[cols])
        acc = accuracy_score(Y_test, Y_pred)
        resultados_n.append({'n': n, 'subconjunto': nombre, 'accuracy': acc})

    tablas[n] = pd.DataFrame(resultados_n)

# una sola tabla con todo
tabla_subconjuntos = pd.concat(tablas.values(), ignore_index=True) 


# Armamos un gráfico para ver accuracy por tamaño de subconjuntos

# GRÁFICO COMPLETO (Y de 0.68 a 1)
fig, ax = plt.subplots()

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Brillantes Cuadrado'],
        label='Brillantes Cuadrado', marker='o')

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Brillantes Casa'],
        label='Brillantes Casa', marker='o')

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Diferencias'],
        label='Diferencias', marker='o')

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Mezcla'],
        label='Mezcla', marker='o')

ax.set_title('Variación del ACC según cantidad de atributos para los distintos subconjuntos, K=5')

ax.set_xlabel('Cantidad de atributos')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.68, 1)
ax.legend()

# GRÁFICO ZOOMEADO (Y de 0.88 a 1)
fig, ax = plt.subplots()

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Brillantes Cuadrado'],
        label='Brillantes Cuadrado', marker='o')

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Brillantes Casa'],
        label='Brillantes Casa', marker='o')

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Diferencias'],
        label='Diferencias', marker='o')

ax.plot('n', 'accuracy', data=tabla_subconjuntos[tabla_subconjuntos['subconjunto'] == 'Mezcla'],
        label='Mezcla', marker='o')

ax.set_title('Variación del ACC según cantidad de atributos para los distintos subconjuntos, K=5')

ax.set_xlabel('Cantidad de atributos')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.88, 1)
ax.legend()

#%% NOS QUEDAMOS CON 255 ATRIBUTOS Y PROBAMOS LAS DIFERENTES COMBINACIONES EN DIFERENTES K 
#El accuracy dio mas alto de todos cuando usamos los subconjuntos de 255 atributos asi que vamos a quedarnos con ese 
# y ver que combinación de atributos y k es la que maximiza la métrica. 

dfp = pixeles_mas_intensos
dfd = diferencias_mas_intensas
nombres_subconjuntos = ['Brillantes Cuadrado', 'Brillantes Casa', 'Diferencias', 'Mezcla']
subconjuntos_255 = generar_subconjunto(255, dfp, dfd)
subconjuntos_9 = generar_subconjunto(9, dfp, dfd)


resultados_dif_k = []  


for k in range(1, 100, 5):
    for nombre, cols in zip(nombres_subconjuntos, subconjuntos_255):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[cols], Y_train)
        Y_pred = knn.predict(X_test[cols])
        acc_train = accuracy_score(Y_train, knn.predict(X_train[cols]))
        acc = accuracy_score(Y_test, Y_pred)
        resultados_dif_k.append({'K': k, 'subconjunto': nombre, 'n': str(cols),
                                 'acc_train': acc_train, 'accuracy': acc})

tabla_resultados_dif_k = pd.DataFrame(resultados_dif_k)
print(tabla_resultados_dif_k) 

# Observamos que siempre la mejor acc por k es la que corresponde al grupo de atributos de la diferencia de las intensidades 
# pero que no mejora linealmente con el aumento de K. En k = 5 tenemos 0.916, lo cual mejora en k = 10 con 0.926 
# pero ya en k = 15 baja levemente en 0.922. 

# Por lo que la mejor combinación resultante fue de hacer 10 vecinos con 10 atributios, los cuales corresponden a los 10 
# pixeles de major diferencia de brillo entre las clases. 

#%% para mirar particularmente un subconjunto con menor cantidad de atributos y ver que no mide mejor el K = 1
resultados_dif_k_9 = []  


for k in range(1, 100, 5):
    for nombre, cols in zip(nombres_subconjuntos, subconjuntos_9):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[cols], Y_train)
        Y_pred = knn.predict(X_test[cols])
        acc_train = accuracy_score(Y_train, knn.predict(X_train[cols]))
        acc = accuracy_score(Y_test, Y_pred)
        resultados_dif_k_9.append({'K': k, 'subconjunto': nombre, 'n': str(cols),
                                 'acc_train': acc_train, 'accuracy': acc})

tabla_resultados_dif_k_9 = pd.DataFrame(resultados_dif_k_9)
print(tabla_resultados_dif_k_9) 

#%% Curva métrica del Acc para diferentes K:

sub = tabla_resultados_dif_k[tabla_resultados_dif_k['subconjunto'] == 'Diferencias']
fig, ax = plt.subplots()

plt.plot(sub['K'], sub['acc_train'], label='Train', marker='o')
plt.plot(sub['K'], sub['accuracy'], label='Test', marker='o')
plt.xlabel('Variación de K vecinos'); plt.ylabel('Accuracy')
plt.title('Variación del ACC según K por Train y Test para el subconjunto "Diferencias"')
plt.legend(); plt.show()


# K mas chico tenes overfitting y el modelo se ajusta casi perfectamente a  cada punto de entrenamiento 
# Un poco antes de K 10 la linea de testeo supera a la del entrenamiemto y en K = 10 consigue su mejor resultado


#%% Curva métrica del Acc para diferentes K en subconjuntos de 9 atributos:

sub_9 = tabla_resultados_dif_k_9[tabla_resultados_dif_k_9['subconjunto'] == 'Diferencias']
fig, ax = plt.subplots()

plt.plot(sub_9['K'], sub_9['acc_train'], label='Train', marker='o')
plt.plot(sub_9['K'], sub_9['accuracy'], label='Test', marker='o')
plt.xlabel('Variación de K vecinos'); plt.ylabel('Accuracy')
plt.title('Variación del ACC según K por Train y Test para el subconjunto "Diferencias" con 9 atributos')
plt.legend(); plt.show()
 

#%% gráficos de lineas para ver variación de ACC según K para las diferentes combinaciones de 10 atributos 

fig, ax = plt.subplots()

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['subconjunto'] == 'Brillantes Cuadrado'],
        label='Brillantes Cuadrado', marker='o')

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['subconjunto'] == 'Brillantes Casa'],
        label='Brillantes Casa', marker='o')

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['subconjunto'] == 'Diferencias'],
        label='Diferencias', marker='o')

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['subconjunto'] == 'Mezcla'],
        label='Mezcla', marker='o')

ax.set_title('Variación del ACC según K para diferentes combinaciones de atributos')

ax.set_xlabel('K')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.8, 1)
ax.legend()

#%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # #                                                                     # #
# #   CLASIFICACIÓN MULTICLASE                                           # #
 # #                                                                     # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#%% Separación entre desarrollo y control
X = imagenes.drop(columns="label")
Y = imagenes["label"]

X_dev , X_held_out, Y_dev, Y_held_out = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

#%% Métricas de evaluación
# Vamos a usar accuracy también ya que vimos en el análisis exploratorio que el conjunto total estaba balanceado 
# y porque al igual que en la parte anterior no hay criterio para priorizar recall o precision por ejemplo 
# también vamos a utilizar la matriz de confusión 

#%%

cv = KFold(n_splits=5, shuffle=True, random_state=42)

resultados_arbol = []

for prof in range(1, 16):
    for crit in ["gini", "entropy"]:
        arbol = DecisionTreeClassifier(criterion=crit, max_depth=prof, random_state=42)
        res = cross_validate(arbol, X_dev, Y_dev, cv=cv,
                             scoring='accuracy', return_train_score=True)
        resultados_arbol.append({
            "profundidad": prof,
            "criterio": crit,
            "acc_train": res["train_score"].mean(),
            "acc_val": res["test_score"].mean()
        })

resultados_arbol = pd.DataFrame(resultados_arbol)

#%% Gráfico que compara los criterios por profunidad

fig, ax = plt.subplots()

ax.plot('profundidad', 'acc_val', data=resultados_arbol[resultados_arbol['criterio'] == 'gini'],
        label='Gini', marker='o')

ax.plot('profundidad', 'acc_val', data=resultados_arbol[resultados_arbol['criterio'] == 'entropy'],
        label='Entropy', marker='o')

ax.plot('profundidad', 'acc_train', data=resultados_arbol[resultados_arbol['criterio'] == 'gini'],
        label='Gini train', marker='o')

ax.plot('profundidad', 'acc_train', data=resultados_arbol[resultados_arbol['criterio'] == 'entropy'],
        label='Entropy train', marker='o')


ax.set_title('Variación del ACC según diferentes profundidades del árbol y distintos criterios por train y test')

ax.set_xlabel('Profundidad')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.16, 1.02)
ax.legend()

#El mejor modelo dentro de las 10 profundidades y los dos criterios fue el de profunidad max (10) con entropia
#Tuvo un desempeño de 0.78 levemente mejor a la misma profundidad pero con gini (0.77)

#%% Miramos matriz de confusion del mejor modelo dentro de los params de la consigna 
mejor_arbol = DecisionTreeClassifier(criterion='entropy', max_depth= 10, random_state= 42)
y_pred_cv = cross_val_predict(mejor_arbol, X_dev, Y_dev, cv=cv)
matriz_conf = confusion_matrix(Y_dev, y_pred_cv, normalize = "true") 

print(matriz_conf)

#%% Hago un heatmap con la matriz de confusion
nombres_clases = clases["nombre"].tolist()

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_conf,
            annot=True, fmt=".2f",
            cmap="Blues",                 
            vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            xticklabels=nombres_clases, yticklabels=nombres_clases,
            cbar_kws={"label": "Proporción de la clase real"})
plt.xlabel("Clase predicha", fontsize = 20)
plt.ylabel("Clase real", fontsize = 20)
plt.title("Matriz de confusión (normalizada por fila)", fontsize = 20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%% Entrenamos el modelo elegido con todo el conjunto de desarrollo

mejor_arbol.fit(X_dev, Y_dev)


#%% Predecir sobre el Held-Out
mejor_arbol_pred = mejor_arbol.predict(X_held_out)
acc_mejor_arbol = accuracy_score(Y_held_out, mejor_arbol_pred)
matriz_held = confusion_matrix(Y_held_out, mejor_arbol_pred, normalize="true")


#%% =============================================================================
# Predecir las clases de las imágenes del conjunto de evaluación  
# ===============================================================================

X_catedra = evaluacion.drop(columns="id")

predict_catedra = mejor_arbol.predict(X_catedra)

predicciones = pd.DataFrame({"id" : evaluacion["id"], "label": predict_catedra})
nombre_archivo = "TP-02-0.7988-GrupoCRV.csv"
predicciones.to_csv(os.path.join(base, nombre_archivo), index=False) # si se abre con excel las clases 0 no aparecen porque excel interpreta que por ej 2,0 es 2
