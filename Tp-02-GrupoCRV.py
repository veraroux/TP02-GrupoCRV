# -*- coding: utf-8 -*-
"""
Materia: Laboratorio de datos - FCEyN - UBA
Comisión A
Autoras  : Aviles Clara, Cabrera Maria Rosario Y Roux Vera
Fecha  : 2026-07-10
Descripción: En el siguiente archivo se encuentran de forma ordenada todas las etapas del Trabajo Práctico N1 de la materia. 
             El mismo está divido por secciones, comenzando con las consultas que realizamos para inspeccionar las tablas originales
             y terminando con la creación de los gráficos pedidos en la consigna. 
             Al correr este archivo se van a generar todas las tablas pedidas, tanto las que corresponden a nuestras Tablas Modelo
             como las que se condicen con las consultas SQL pedidas. También se van a generar todos los gráficos que utilizamos a lo largo 
             de todo nuestro análisis, los cuales se pueden encontrar en la carpeta denominada de la misma forma. 
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


SEMILLA = 42    


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

# Plot imagen
img = np.array(X.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

img = np.array(X.iloc[2]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()


img = np.array(X.iloc[56]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

img = np.array(X.iloc[105]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()


#%% Por Clase, pixeles promeido de mayor intesidad: 

#%%Imágen promedio por clase 

fig, axes = plt.subplots(2, 5, figsize = (20, 10))
for c, ax in zip(range(10), axes.ravel()): 
    media = X[Y == c].mean().values.reshape(28,28)
    ax.imshow(media, cmap='gray')
    ax.set_title(f'Clase {c}', fontsize = 20)
    ax.tick_params(axis='both', labelsize=16)

plt.show()


#%% pixeles menos intesos: 

dic_valores = {}
for c in range (10): 
    dic_valores[f'Clase {c}'] = X[Y== c].mean().sort_values().head(10)

pixeles_menos_intesos = pd.DataFrame(dic_valores)

print(pixeles_menos_intesos)

#%% pixeles mas intesos: 

dic_valores = {}
for c in range (10): 
    dic_valores[f'Clase {clases.loc[c, "nombre"]}'] = X[Y== c].mean().sort_values(ascending = False).head(10)

pixeles_mas_intesos = pd.DataFrame(dic_valores)

print(pixeles_mas_intesos)

#%% Pixeles entre clases similares ver intensidades que diferencien: 
    

#%% Diferencias entre 0(circulo) y 7 (paraguas)

dif_0_7 = (X[Y == 0].mean() - X[Y == 7].mean()).abs()
print(dif_0_7)


img = np.array(dif_0_7).reshape((28,28))
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
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
plt.imshow(img, cmap='gray')
plt.show()

"""
ANALISIS: 
"""
#%%

dic_diferencias = {}
for c in range (10): 
    for i in range(c+1, 10): 
        
        dic_diferencias[f'Clase {clases.loc[c, "nombre"]} - Clase {clases.loc[i, "nombre"]}'] =  (X[Y == c].mean() - X[Y == i].mean()).abs().nlargest(10)

diferencias_mas_intesas = pd.DataFrame(dic_diferencias)

print(diferencias_mas_intesas)



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

X = df_clas_binaria.drop(columns="label")
Y = df_clas_binaria["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

#%% =============================================================================
# Ajustar el KNN con diferentes cantidades de atributos   
# ===============================================================================
 
#%% UN K FIJO, CONJUNTOS DE 3 ATRIBUTOS     
k = 5

# Pixeles mas brillantes de la clase 2: del 40 al 49
# Pixeles mas brillantes de la clase 9: 225, 282, 311, 283, 365, 393, 394, 283, 366, 310, 338
#Pixeles de mayor diferencia de intensidad entre las clases: 51, 113, 169, 141,45, 50, 46, 49, 47, 48
#La mayoría de los pixeles que tienen mayor diferencia se encuentran entre los mas brillantes de la clase 2
#por lo que tambien deben ser de los menos brillantes en la clase 9. 

subconjuntos_3 = [["pixel45", "pixel46", "pixel47"], ["pixel338", "pixel310", "pixel366"], 
                ["pixel47", "pixel48", "pixel49"], ["pixel46", "pixel338", "pixel48"]]

resultados_3 = []

for cols in subconjuntos_3:
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train[cols], Y_train)
    Y_pred = knn.predict(X_test[cols])
    acc = accuracy_score(Y_test, Y_pred)
 
    
    print(f"atributos={cols}: exactitud={acc:.3f}")
    resultados_3.append({'atributos': str(cols), 'accuracy': acc})

tabla_resultados_3 = pd.DataFrame(resultados_3)
print(tabla_resultados_3)



#%% UN K FIJO, CONJUNTOS DE 6 ATRIBUTOS     

k = 5

subconjuntos_5 = [["pixel45", "pixel46", "pixel47", "pixel44", "pixel48", "pixel43"], 
                ["pixel338", "pixel310", "pixel366", "pixel283", "pixel394", "pixel365"], 
                ["pixel47", "pixel48", "pixel49", "pixel46", "pixel50", "pixel45"], 
                ["pixel46","pixel45" ,"pixel338", "pixel310", "pixel48", "pixel47"]]
resultados_6 = []

for cols in subconjuntos_5:
    knn = KNeighborsClassifier(n_neighbors= k)
    knn .fit(X_train[cols], Y_train)
    Y_pred = knn.predict(X_test[cols])
    acc = accuracy_score(Y_test, Y_pred)

    
    print(f"atributos={cols}: exactitud={acc:.3f}")
    resultados_6.append({'atributos': str(cols), 'accuracy': acc})

tabla_resultados_6 = pd.DataFrame(resultados_6)
print(tabla_resultados_6)

#%% UN K FIJO, CONJUNTOS DE 10 ATRIBUTOS     

k = 5

subconjuntos_10 = [["pixel40", "pixel41", "pixel42", "pixel49", "pixel45", "pixel46", "pixel47", "pixel44", "pixel48", "pixel43"], 
                ["pixel225", "pixel282", "pixel311", "pixel283", "pixel365", "pixel393", "pixel394", "pixel366", "pixel310", "pixel338"], 
                ["pixel51", "pixel113", "pixel169", "pixel141", "pixel45", "pixel50", "pixel46", "pixel49", "pixel47", "pixel48"], 
                ["pixel46","pixel49", "pixel48" ,"pixel338", "pixel310", "pixel47", "pixel40", "pixel282", "pixel311", "pixel283"]]
resultados_10 = []

for cols in subconjuntos_10:
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train[cols], Y_train)
    Y_pred = knn.predict(X_test[cols])
    acc = accuracy_score(Y_test, Y_pred)

    
    print(f"atributos={cols}: exactitud={acc:.3f}")
    resultados_10.append({'atributos': str(cols), 'accuracy': acc})

tabla_resultados_10 = pd.DataFrame(resultados_10)
print(tabla_resultados_10)




#%% NOS QUEDAMOS CON 10 ATRIBUTOS Y PROBAMOS LAS DIFERENTES COMBINACIONES EN DIFERENTES K 
#El accuracy dio mas alto de todos cuando usamos los subconjuntos de 10 atributos asi que vamos a quedarnos con ese 
# y ver que combinación de atributos y k es la que maximiza la métrica. 

nombres = ['Brillantes Cuadrado', 'Brillantes Casa', 'Diferencias', 'Mezcla']

resultados_dif_k = []  

for k in [3,5,10,15,30]:
    for nombre, cols in zip(nombres, subconjuntos_10):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[cols], Y_train)
        Y_pred = knn.predict(X_test[cols])
        acc_train = accuracy_score(Y_train, knn.predict(X_train[cols]))
        acc = accuracy_score(Y_test, Y_pred)
        resultados_dif_k.append({'K': k, 'nombre': nombre, 'atributos': str(cols),
                                 'acc_train': acc_train, 'accuracy': acc})

tabla_resultados_dif_k = pd.DataFrame(resultados_dif_k)
print(tabla_resultados_dif_k) 

# Observamos que siempre la mejor acc por k es la que corresponde al grupo de atributos de la diferencia de las intensidades 
# pero que no mejora linealmente con el aumento de K. En k = 5 tenemos 0.916, lo cual mejora en k = 10 con 0.926 
# pero ya en k = 15 baja levemente en 0.922. 

# Por lo que la mejor combinación resultante fue de hacer 10 vecinos con 10 atributios, los cuales corresponden a los 10 
# pixeles de major diferencia de brillo entre las clases. 

#%% Curva métrica del Acc para diferentes K:

sub = tabla_resultados_dif_k[tabla_resultados_dif_k['nombre'] == 'Diferencias']

plt.plot(sub['K'], sub['acc_train'], label='Train', marker='o')
plt.plot(sub['K'], sub['accuracy'], label='Test', marker='o')
plt.xlabel('Variación de K vecinos'); plt.ylabel('Accuracy')
plt.legend(); plt.show()


# K mas chico tenes overfitting y el modelo se ajusta casi perfectamente a  cada punto de entrenamiento 
# Un poco antes de K 10 la linea de testeo supera a la del entrenamiemto y en K = 10 consigue su mejor resultado 

#%% gráficos de lineas para ver variación de ACC según K para las diferentes combinaciones de 10 atributos 

fig, ax = plt.subplots()

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['nombre'] == 'Brillantes Cuadrado'],
        label='Brillantes Cuadrado', marker='o')

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['nombre'] == 'Brillantes Casa'],
        label='Brillantes Casa', marker='o')

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['nombre'] == 'Diferencias'],
        label='Diferencias', marker='o')

ax.plot('K', 'accuracy', data=tabla_resultados_dif_k[tabla_resultados_dif_k['nombre'] == 'Mezcla'],
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

for prof in range (1, 11): 
    for crit in ["gini", "entropy"]:
        arbol = DecisionTreeClassifier(criterion=crit, max_depth=prof, random_state=42)
        scores = cross_val_score(arbol, X_dev, Y_dev, cv=cv, scoring = 'accuracy')
        resultados_arbol.append({"profundidad": prof, "criterio": crit, "acc_media": scores.mean()})

resultados_arbol = pd.DataFrame(resultados_arbol)

#%% Gráfico que compara los criterios por profunidad


fig, ax = plt.subplots()

ax.plot('profundidad', 'acc_media', data=resultados_arbol[resultados_arbol['criterio'] == 'gini'],
        label='Gini', marker='o')

ax.plot('profundidad', 'acc_media', data=resultados_arbol[resultados_arbol['criterio'] == 'entropy'],
        label='Entropy', marker='o')



ax.set_title('Variación del ACC según diferentes profundidades del árbol y distintos criterios')

ax.set_xlabel('Profundidad')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.legend()

#El mejor modelo dentro de las 10 profundidades y los dos criterios fue el de profunidad max (10) con entropia
#Tuvo un desempeño de 0.78 levemente mejor a la misma profundidad pero con gini (0.77)

#%% Miramos matriz de confusion del mejor modelo 
mejor_arbol = DecisionTreeClassifier(criterion='entropy', max_depth= 10, random_state= 42)
y_pred_cv = cross_val_predict(mejor_arbol, X_dev, Y_dev, cv=cv)
matriz_conf = confusion_matrix(Y_dev, y_pred_cv, normalize = "true") 

print(matriz_conf)

#%% Hago un heatmap con la matriz de confusion
nombres = ["círculo", "escalera", "cuadrado", "pez", "triángulo",
           "colectivo", "estrella", "paraguas", "remera", "casa"]

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_conf,
            annot=True, fmt=".2f",
            cmap="Blues",                 
            vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            xticklabels=nombres, yticklabels=nombres,
            cbar_kws={"label": "Proporción de la clase real"})
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.title("Matriz de confusión (normalizada por fila)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%% Entrenamos el modelo elegido con todo el conjunto de desarrollo

mejor_arbol.fit(X_dev, Y_dev)

#%% Predecir sobre el Held-Out
