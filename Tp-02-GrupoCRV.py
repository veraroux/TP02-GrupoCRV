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