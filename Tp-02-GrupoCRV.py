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

#%% Cantidad de imágenes por clase: 

cant_img_clase = """
                SELECT c.nombre, COUNT(*) AS cant,
                FROM clases as c 
                INNER JOIN imagenes as i
                    ON c.label = i.label 
                GROUP BY c.nombre 
                """

df_img_clases = dd.sql(cant_img_clase).df()
print(df_img_clases)



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

fig, axes = plt.subplots(2, 5)
for c, ax in zip(range(10), axes.ravel()): 
    media = X[Y == c].mean().values.reshape(28,28)
    ax.imshow(media, cmap='gray')
    ax.set_title(f'Clase {c}')
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
    dic_valores[f'Clase {c}'] = X[Y== c].mean().sort_values(ascending = False).head(10)

pixeles_mas_intesos = pd.DataFrame(dic_valores)

print(pixeles_mas_intesos)

#%% Pixeles entre clases similares ver intensidades que diferencien: 

#%% Diferencias entre 0(circulo) y 7 (paraguas)

dif_0_7 = (X[Y == 0].mean() - X[Y == 7].mean()).abs()
print(dif_0_7)


img = np.array(dif_0_7).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()


#%% Diferencias entre 0(circulo) y 2 (cuadrado)

dif_0_2 = (X[Y == 0].mean() - X[Y == 2].mean()).abs()
print(dif_0_2)

img = np.array(dif_0_2).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

