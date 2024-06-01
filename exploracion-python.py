from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

data = datasets.load_iris()
print(data)

data = pd.DataFrame(datasets.load_iris().data, 
    columns = ["Largo Sepalo", "Ancho sepalo", "Largo Petalo", "Ancho Petalo"])

data = pd.read_csv("Iris.csv")

#imprimir el objeto de datos
print(data)

#devuelve la forma (filas y columnas del data frame)
print(data.shape)

#Devuelve los nombres de las columnas
print(data.columns)

#Informacion general sobre el conjunto de datos
print(data.info)

#Generación de los estadisticos básicos
print(data.describe())

#Generación de los estadisticos básicos para 1 columna del data frame
print(data['SepalLengthCm'].describe())

#devuelve los primeros 5 registros
print(data.head())

#Devuelve los primeros 25 registros
print(data.head(25))

#Devuelve los ultimos 5 registros
print(data.tail())

#Devuelve los ultimos 25 registros
print(data.tail(25))

#Devuelve true para aquellos atributos que no no sean numericos o nulos
print(data.isna())

#Devuelve true para aquellos atributos quesean nulos
print(data.isnull())

#Devuelve el tamano (cantidad de registros) para un objeto de datos
#En este caso se agrupa por especies
print(data.groupby('Species').size())
print(data['Species'].value_counts())

#Completa con el argumento que le pasemos los valores nulos
data.fillna("")

#Crear un subconjunto de datos a partir del conjunto principal para los valores seleccionados
setosa = data[data['Species'] == 'Iris-setosa']
versicolor = data[data['Species'] == 'Iris-versicolor']
virginica = data[data['Species'] == 'Iris-virginica']


#Crear un subplot para agrupar graficos
x1 = plt.subplot()

#Crear los graficos de tipo dispersion en funcion de las variables x,y
g_setosa = setosa.plot(x = 'SepalLengthCm', y = 'SepalWidthCm', kind = 'scatter', color = 'red', ax = x1)
g_versicolor = versicolor.plot(x = 'SepalLengthCm', y = 'SepalWidthCm', kind = 'scatter', color = 'green', ax = x1)
g_virginica = virginica.plot(x = 'SepalLengthCm', y = 'SepalWidthCm', kind = 'scatter', color = 'blue', ax = x1)

#muestra el grafico en pantalla
plt.show()


#Hacer un grafico de barras: ganado y perdido tienen la data a graficar
ganado = [1, 17.5, 40, 48, 52, 69, 88]
perdido = [2, 8, 70, 1.5, 25, 12, 28]
index = ['Equipo A', 'Equipo B', 'Equipo C',
         'Equipo D', 'Equifo E', 'Equipo F', 'Equipo G']
df = pd.DataFrame({'Ganado': ganado,
                   'Perdido': perdido}, index=index)
ax = df.plot.bar(rot=0)
ax = df.plot.bar(rot = 0, stacked = True)
ax = df.plot.bar(rot = 0, subplots = True)

plt.show()