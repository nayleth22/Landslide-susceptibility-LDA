from osgeo import gdal
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import shapiro, bartlett
from matplotlib import pyplot as plt
import argparse


# define los argumentos de entrada del script
parser = argparse.ArgumentParser(description='Calcula el LDA')
# tiene un numero n de argumentos de entrada donde cada entrada es una direccion de archivo
parser.add_argument('--t-maps', metavar='file', type=str, nargs='+',
                    help='Mapas tematicos, deben tener el mismo tamaño y el mismo tamaño de pixel' , required=True)
parser.add_argument('--inv_train', metavar='file', type=str, nargs=1, 
                    help='Inventario de entrenamiento de Procesos morfodinamicos, deben tener el mismo tamaño y el mismo tamaño de pixel', required=True)
parser.add_argument('--inv_test', metavar='file', type=str, nargs=1, 
                    help='Inventario de testeo de Procesos morfodinamicos, deben tener el mismo tamaño y el mismo tamaño de pixel', required=True)
parser.add_argument("--null", metavar='float', type=float, nargs=1,
                    help='Valor nulo de los mapas si este existe, debe ser el mismo para todos, por defecto es -1', default=-1)
parser.add_argument("--out", metavar='dir', type=str, nargs=1, required=True,
                    help='Directorio de salida')
args = parser.parse_args()

# define las variables de entrada
t_maps = args.t_maps
inv_train = args.inv_train[0]
inv_test = args.inv_test[0]
null = args.null
out = args.out[0]

# lee los mapas tematicos
t_maps_ds = []
for t_map in t_maps:
    t_maps_ds.append(gdal.Open(t_map))

# lee el inventario de procesos morfodinamicos
inv_train_ds = gdal.Open(inv_train)
inv_test_ds = gdal.Open(inv_test)


# crea un archivo txt para guardar los resultados
out_txt = open(out + "lda.txt", "w")

# imprime los argumentos de entrada en el archivo txt
out_txt.write("Argumentos de entrada: \n")
out_txt.write("Mapas tematicos: " + str(t_maps) + "\n")
out_txt.write("Inventario de entrenamiento de procesos morfodinamicos: " + str(inv_train) + "\n")
out_txt.write("Inventario de testeo de procesos morfodinamicos: " + str(inv_test) + "\n")
out_txt.write("Valor nulo: " + str(null) + "\n")
out_txt.write("Directorio de salida: " + str(out) + "\n")
out_txt.write("\n")



# crea una matriz para guardar las dimensiones de los mapas tematicos
t_maps_shape = np.zeros((len(t_maps_ds), 2), dtype=int)

# obten las dimensiones de todos los raster de t_maps e imprime las dimensiones
for t_map_ds in t_maps_ds:
    t_map_ds_band = t_map_ds.GetRasterBand(1)
    t_map_ds_band_array = t_map_ds_band.ReadAsArray()
    # imprime las dimensiones de la matriz de cada raster de t_maps con su nombre
    #print(t_map_ds.GetDescription(), t_map_ds_band_array.shape)
    # guarda las dimensiones de la matriz de cada raster de t_maps
    t_maps_shape[t_maps_ds.index(t_map_ds), 0] = t_map_ds_band_array.shape[0]
    t_maps_shape[t_maps_ds.index(t_map_ds), 1] = t_map_ds_band_array.shape[1]

# obtiene las dimensiones del raster de inv
inv_train_ds_band = inv_train_ds.GetRasterBand(1)
inv_train_ds_band_array = inv_train_ds_band.ReadAsArray()
inv_test_ds_band = inv_test_ds.GetRasterBand(1)
inv_test_ds_band_array = inv_test_ds_band.ReadAsArray()
# imprime las dimensiones de la matriz de inv
#print(inv_ds.GetDescription(), inv_ds_band_array.shape)

# guarda las dimensiones de la matriz de inv en t_maps_shape usando append
t_maps_shape = np.append(t_maps_shape, np.array([[inv_train_ds_band_array.shape[0], inv_train_ds_band_array.shape[1]]]), axis=0)

# valida que todas las matrices tengan las mismas dimensiones y el mismo tamaño de pixel
# si no es asi, termina el script
# si es asi, solo continua
if np.all(t_maps_shape == t_maps_shape[0]):
    print("Las dimensiones de los mapas son iguales")
else:
    print("Las dimensiones de los mapas no son iguales")
    exit()

# crea un frame de pandas para guardar los datos de los mapas tematicos
t_maps_df = pd.DataFrame()

# crea una lista para guardar los nombres de las columnas
t_maps_df_columns = []


for t_map_ds in t_maps_ds:
    t_map_ds_band = t_map_ds.GetRasterBand(1)
    t_map_ds_band_array = t_map_ds_band.ReadAsArray()
    # crea una lista para guardar los valores de cada pixel de cada raster de t_maps
    t_map_ds_band_array_list = []
    # recorre la matriz de cada raster de t_maps
    for i in range(t_map_ds_band_array.shape[0]):
        for j in range(t_map_ds_band_array.shape[1]):
            t_map_ds_band_array_list.append(t_map_ds_band_array[i, j])
    #obten el nombre del archivo despues del ultimo slash
    t_map_ds_name = t_map_ds.GetDescription().split("/")[-1]
    # dejalo solo con 10 caracteres si es mas largo
    if len(t_map_ds_name) > 10:
        t_map_ds_name = t_map_ds_name[:10]
    
    # crea una columna en el frame de pandas con los valores de cada pixel de cada raster de t_maps
    t_maps_df[t_map_ds_name] = t_map_ds_band_array_list
    # agrega el nombre de la columna a la lista de nombres de columnas
    t_maps_df_columns.append(t_map_ds_name)

t_maps_train_df = t_maps_df.copy()

t_maps_test_df = t_maps_df.copy()
#crea una nueva columna en el dataframe llamada clases y asigna  los valores presentes en inv_train_ds_band_array o inv_test_ds_band_array segun corresponda.
t_maps_train_df["clases"] = np.reshape(np.ravel(inv_train_ds_band_array), (inv_train_ds_band_array.shape[0] * inv_train_ds_band_array.shape[1], 1))
t_maps_test_df["clases"] = np.reshape(np.ravel(inv_test_ds_band_array), (inv_test_ds_band_array.shape[0] * inv_test_ds_band_array.shape[1], 1))

t_maps_train_df.columns = t_maps_df_columns + ["clases"]
t_maps_test_df.columns = t_maps_df_columns + ["clases"]
#reemplaza los valores nulos por el valor NaN (Not a Number) de numpy. en los dataframes originales
t_maps_train_df.replace(null, np.nan, inplace=True)
t_maps_test_df.replace(null, np.nan, inplace=True)
t_maps_df.replace(null, np.nan, inplace=True)

# crea x train y test
#x_train, x_test, y_train, y_test = train_test_split(t_maps_df[t_maps_df_columns], t_maps_df["clases"], test_size=test_percent, stratify=t_maps_df["clases"],)
x_train = t_maps_train_df[t_maps_df_columns]
x_test = t_maps_test_df[t_maps_df_columns]
y_train = t_maps_train_df["clases"]
y_test = t_maps_test_df["clases"]


train_df = pd.concat([x_train, y_train], axis=1)
test_df = pd.concat([x_test, y_test], axis=1)

# eliminar las filas con valores nulos
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

for column in train_df.columns[:-1]:
    # x es la muestra de la clase 1
    stat, p = shapiro(train_df[column])

    # Imprimir el valor p de la prueba
    print("Valor p de la prueba de Shapiro-Wilk para la clase {}: {}".format(column, p))
    out_txt.write("Valor p de la prueba de Shapiro-Wilk para la clase {}: {}".format(column, p) + "\n")

out_txt.write("\n")

# Realizar la prueba de Bartlett
stat, p = bartlett(*[train_df[col] for col in train_df.columns[:-1]])

# Imprimir los resultados
print("Estadístico de prueba Bartlett:", stat)
print("Valor p:", p)
out_txt.write("Estadístico de prueba Bartlett: {}".format(stat) + "\n")
out_txt.write("Valor p: {}".format(p) + "\n\n")

#realizar la prueba de bartlett para los que tienen deslizamiento
train_df_slip = train_df[train_df["clases"] == 1]
stat, p = bartlett(*[train_df_slip[col] for col in train_df_slip.columns[:-1]])

# Imprimir los resultados
print("Estadístico de prueba Bartlett para los que tienen deslizamiento:", stat)
print("Valor p:", p)
out_txt.write("Estadístico de prueba Bartlett para los que tienen deslizamiento: {}".format(stat) + "\n")
out_txt.write("Valor p: {}".format(p) + "\n\n")

#realizar la prueba de bartlett para los que no tienen deslizamiento
train_df_no_slip = train_df[train_df["clases"] == 0]
stat, p = bartlett(*[train_df_no_slip[col] for col in train_df_no_slip.columns[:-1]])

# Imprimir los resultados
print("Estadístico de prueba Bartlett para los que no tienen deslizamiento:", stat)
print("Valor p:", p)
out_txt.write("Estadístico de prueba Bartlett para los que no tienen deslizamiento: {}".format(stat) + "\n")
out_txt.write("Valor p: {}".format(p) + "\n\n")


corr_matrix = x_train.corr()
out_txt.write("Matriz de correlacion: \n{}".format(corr_matrix) + "\n\n")

# dibuja la matriz de correlacion
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Matriz de correlacion')
plt.tight_layout()
#guarda la figura en un archivo png en la direccion out en muy buena calidad
plt.savefig(out + "matriz_correlacion.png", dpi=300)
#plt.show()
# limpia la figura
plt.clf()


# calcular la matriz de covarianza para cuando hay deslizamiento
cov_matrix_slip = np.cov(train_df_slip[:-1].values, rowvar=False)
print("Matriz de covarianza para cuando hay deslizamiento: \n{}".format(cov_matrix_slip))
out_txt.write("Matriz de covarianza para cuando hay deslizamiento: {}".format(cov_matrix_slip) + "\n")

# verifica que la matriz de covarianza es la misma para todas las clases cuando hay deslizamiento
print("las matrices de covarianza son iguales para todas las clases cuando hay deslizamiento: {}".format(np.allclose(cov_matrix_slip, cov_matrix_slip.T)))
out_txt.write("las matrices de covarianza son iguales para todas las clases cuando hay deslizamiento: {}".format(np.allclose(cov_matrix_slip, cov_matrix_slip.T)) + "\n")
out_txt.write("\n")

plt.figure(figsize=(10, 10))
sns.heatmap(cov_matrix_slip, annot=True, cmap='coolwarm', xticklabels=t_maps_df_columns, yticklabels=t_maps_df_columns, )
plt.xticks(rotation=0)
plt.title('Matriz de covarianza para cuando hay deslizamiento')
# estabelce tinght layout
plt.tight_layout()
#plt.show()
#guarda la figura en un archivo png en la direccion out
plt.savefig(out + "matriz_covarianza_slip.png", dpi=300)
plt.clf()

# calcular la matriz de covarianza para cuando no hay deslizamiento
cov_matrix_no_slip = np.cov(train_df_no_slip[:-1].values, rowvar=False)
print("Matriz de covarianza para cuando no hay deslizamiento: \n{}".format(cov_matrix_no_slip))
out_txt.write("Matriz de covarianza para cuando no hay deslizamiento: {}".format(cov_matrix_no_slip) + "\n")

# verifica que la matriz de covarianza es la misma para todas las clases cuando no hay deslizamiento
print("las matrices de covarianza son iguales para todas las clases cuando no hay deslizamiento: {}".format(np.allclose(cov_matrix_no_slip, cov_matrix_no_slip.T)))
out_txt.write("las matrices de covarianza son iguales para todas las clases cuando no hay deslizamiento: {}".format(np.allclose(cov_matrix_no_slip, cov_matrix_no_slip.T)) + "\n")
out_txt.write("\n")

plt.figure(figsize=(10, 10))
sns.heatmap(cov_matrix_no_slip, annot=True, cmap='coolwarm', xticklabels=t_maps_df_columns, yticklabels=t_maps_df_columns, )
plt.xticks(rotation=0)
plt.title('Matriz de covarianza para cuando no hay deslizamiento')
# estabelce tinght layout
plt.tight_layout()
#plt.show()
#guarda la figura en un archivo png en la direccion out
plt.savefig(out + "matriz_covarianza_no_slip.png", dpi=300)
plt.clf()


# cuenta el número de 1s y 0s en y_train
counts = train_df['clases'].value_counts()

# calcula el tamaño de la muestra deseada para cada clase
desired_size = min(counts)

# submuestrea el DataFrame seleccionando una muestra aleatoria fija de filas para cada clase
df_balanced = pd.concat([train_df[train_df['clases'] == 0].sample(n=desired_size, random_state=42),
                         train_df[train_df['clases'] == 1].sample(n=desired_size, random_state=42)])

# mezcla el nuevo DataFrame balanceado
train_df = df_balanced.sample(frac=1, random_state=42)

x_train = train_df[t_maps_df_columns]
y_train = train_df["clases"]
x_test = test_df[t_maps_df_columns]
y_test = test_df["clases"]

#normaliza cada columna de xtrain
mean_std = np.zeros((len(t_maps_df_columns), 2), dtype=float)

for i in range(len(t_maps_df_columns)):
    mean_std[i, 0] = np.mean(x_train[t_maps_df_columns[i]])
    mean_std[i, 1] = np.std(x_train[t_maps_df_columns[i]])
    x_train[t_maps_df_columns[i]] = (x_train[t_maps_df_columns[i]] - mean_std[i, 0] ) / (mean_std[i, 1])

# normaliza cada columna de xtest
for i in range(len(t_maps_df_columns)):
    x_test[t_maps_df_columns[i]] = (x_test[t_maps_df_columns[i]] - mean_std[i, 0] ) / (mean_std[i, 1])

print("media, desviacion estandar:\n", mean_std)
out_txt.write("media, desviacion estandar:\n{}".format(mean_std) + "\n")
out_txt.write("\n")

# LDA
sklearn_lda = LDA(n_components=1, solver='eigen', shrinkage='auto', priors=None, store_covariance=False,)
X_lda_sklearn = sklearn_lda.fit_transform(x_train, y_train)

print('Eigenvalues: ', sklearn_lda.explained_variance_ratio_)
print('Components: ', sklearn_lda.coef_)
out_txt.write('Eigenvalues: {}'.format(sklearn_lda.explained_variance_ratio_) + "\n\n")
out_txt.write('Components: {}'.format(sklearn_lda.coef_) + "\n\n")

# imprime el nombre de X y su coef
print('X: ', t_maps_df.columns[:-1] )
print('Coef: ', sklearn_lda.coef_)
print('scaler: ', sklearn_lda.scalings_)
print('classes: ', sklearn_lda.classes_)
print('n_features: ', sklearn_lda.n_features_in_)
print('intercept: ', sklearn_lda.intercept_)
out_txt.write('X: {}'.format(t_maps_df.columns[:-1]) + "\n\n")
out_txt.write('Coef: {}'.format(sklearn_lda.coef_) + "\n\n")
out_txt.write('scaler: {}'.format(sklearn_lda.scalings_) + "\n\n")
out_txt.write('classes: {}'.format(sklearn_lda.classes_) + "\n\n")
out_txt.write('n_features: {}'.format(sklearn_lda.n_features_in_) + "\n\n")
out_txt.write('intercept: {}'.format(sklearn_lda.intercept_) + "\n\n")


parametros = sklearn_lda.get_params()
print('Parametros: ', parametros)
out_txt.write('Parametros: {}'.format(parametros) + "\n\n")
columns_names = sklearn_lda.get_feature_names_out()

# imprime los parametros usados para la lda  generada
print('LDA de scikit-learn: ', sklearn_lda)
out_txt.write('LDA de scikit-learn: {}'.format(sklearn_lda) + "\n\n")

coeficientes = sklearn_lda.coef_[0]
intercepto = sklearn_lda.intercept_

# imprimir la ecuación
ecuacion = "y = {:.5e}".format(intercepto[0])
for i, coef in enumerate(coeficientes):
    ecuacion += " + {:.5e}x{}".format(coef, i+1)
print(ecuacion)
out_txt.write(ecuacion + "\n\n")

# grafica la ROC train
y_pred_prob = sklearn_lda.predict_proba(x_train)[:, 1]
fpr, tpr, umbrales = roc_curve(y_train, y_pred_prob)
auc_roc = roc_auc_score(y_train, y_pred_prob)
# obten cada percentil de los datos de test con su respectivo valor
percentiles = np.percentile(y_pred_prob, np.arange(0, 100, 1))

# escribelos los percentiles frente a su percentil
out_txt.write('Percentiles Train: ' + "\n")
for i in range(len(percentiles)):
    out_txt.write("{:.5f} {:.5f}".format(i, percentiles[i]) + "\n")
out_txt.write("\n")

print('AUC ROC: ', auc_roc)
print('fpr , tpr, umbrales: ', fpr, tpr, umbrales)
out_txt.write('AUC ROC Train: {}'.format(auc_roc) + "\n")
# imprime los valores de fpr, tpr y umbrales uno frente al otro como una tabla
out_txt.write('fpr , tpr, umbrales: ' + "\n")
for i in range(len(fpr)):
    out_txt.write("{:.5f} {:.5f} {:.5f}".format(fpr[i], tpr[i], umbrales[i]) + "\n")


out_txt.write("\n")

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.3f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC Train')
plt.legend(loc="lower right")
plt.tight_layout()
#plt.show()
#guarda la figura en un archivo png en la direccion out
plt.savefig(out + "Curva_ROC_Train.png", dpi=300)
plt.clf()

# grafica la ROC test
y_pred_prob = sklearn_lda.predict_proba(x_test)[:, 1]
fpr, tpr, umbrales = roc_curve(y_test, y_pred_prob)
auc_roc = roc_auc_score(y_test, y_pred_prob)
# obten cada percentil de los datos de test con su respectivo valor
percentiles = np.percentile(y_pred_prob, np.arange(0, 100, 1))


out_txt.write('Percentiles Test: ' + "\n")
for i in range(len(percentiles)):
    out_txt.write("{:.5f} {:.5f}".format(i, percentiles[i]) + "\n")
out_txt.write("\n")


print('AUC ROC: ', auc_roc)
print('fpr , tpr, umbrales: ', fpr, tpr, umbrales)
out_txt.write('AUC ROC Test: {}'.format(auc_roc) + "\n")
# imprime los valores de fpr, tpr y umbrales uno frente al otro como una tabla
out_txt.write('fpr , tpr, umbrales: ' + "\n")
for i in range(len(fpr)):
    out_txt.write("{:.5f} {:.5f} {:.5f}".format(fpr[i], tpr[i], umbrales[i]) + "\n")

out_txt.write("\n")

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.3f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC Test')
plt.legend(loc="lower right")
plt.tight_layout()
#plt.show()
#guarda la figura en un archivo png en la direccion out
plt.savefig(out + "Curva_ROC_Test.png", dpi=300)
plt.clf()


# grafica la matriz de confusión
y_pred = sklearn_lda.predict(x_test)
matriz_confusion = confusion_matrix(y_test, y_pred)
print('Matriz de confusión: ')
print(matriz_confusion)
out_txt.write('Matriz de confusión: \n{}'.format(matriz_confusion) + "\n")

plt.figure(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=sklearn_lda.classes_)
disp.plot()

#plt.show()
#guarda la figura en un archivo png en la direccion out
plt.savefig(out + "Matriz_Confusion.png", dpi=300)
plt.clf()

y_pred = sklearn_lda.predict(x_train) # Obtener las predicciones

# Extraer los datos y las etiquetas
x_embb = x_train.iloc[:, :-1].values

# Realizar una reducción de dimensionalidad con PCA a 3 dimensiones
pca = PCA(n_components=3)
X_pca = pca.fit_transform(x_train)


# Crear una figura en 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotear los puntos en 3D según su reducción de dimensionalidad
for label in set(y_train):
    if label == 0:
        ax.scatter(X_pca[y_train==label, 0], X_pca[y_train==label, 1], X_pca[y_train==label, 2],  label='No deslizamiento',
                   marker='o', facecolors='none', edgecolors='blue')
    else:
        ax.scatter(X_pca[y_train==label, 0], X_pca[y_train==label, 1], X_pca[y_train==label, 2],  label='Deslizamiento',
                   marker='o', facecolors='none', edgecolors='red')

# Configurar los ejes y la leyenda
ax.set_xlabel('AXIS 1')
ax.set_ylabel('AXIS 2')
ax.set_zlabel('AXIS 3')
ax.legend()
#agrega el titulo
ax.set_title('Antes de la LDA')
fig.tight_layout()

#plt.show()
#guarda la figura en un archivo png en la direccion out
fig.savefig(out + "Antes_LDA.png", dpi=300)
plt.clf()

# Crear una figura en 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotear los puntos en 3D según su reducción de dimensionalidad
for label in set(y_train):
    if label == 0:
        ax.scatter(X_pca[y_pred==label, 0], X_pca[y_pred==label, 1], X_pca[y_pred==label, 2],  label='No deslizamiento',
                   marker='o', facecolors='none', edgecolors='blue')
    else:
        ax.scatter(X_pca[y_pred==label, 0], X_pca[y_pred==label, 1], X_pca[y_pred==label, 2],  label='Deslizamiento',
                   marker='o', facecolors='none', edgecolors='red')

# Configurar los ejes y la leyenda
ax.set_xlabel('AXIS 1')
ax.set_ylabel('AXIS 2')
ax.set_zlabel('AXIS 3')
ax.legend()
#agrega el titulo
ax.set_title('Despues de la LDA')
fig.tight_layout()

#plt.show()
#guarda la figura en un archivo png en la direccion out
fig.savefig(out + "Despues_LDA.png", dpi=300)
plt.clf()

n_null = np.min(t_maps_df.values) - 9999
#encontrar índices con valores NaN
nan_indices = t_maps_df.index[t_maps_df.isnull().any(axis=1)].tolist()

t_maps_df.replace(np.nan, n_null, inplace=True)

# normalizar los datos de maps por columna
for i in range(len(t_maps_df_columns)):
        t_maps_df[t_maps_df_columns[i]] = (t_maps_df[t_maps_df_columns[i]] - mean_std[i, 0] ) / (mean_std[i, 1])



maps_list = []

#predice cada pixel de la matriz de salida

# eliminar filas con NaN
t_maps_df = t_maps_df.dropna()
maps_list = sklearn_lda.predict_proba(t_maps_df.iloc[:, :])[::,1]
# agregar NaNs en los índices originales de NaNs
for i in nan_indices:
    maps_list = np.insert(maps_list, i, np.nan)

'''load = 0
max_load = t_maps_df.shape[0]
print("Prediciendo valores de la matriz de salida...")
for i in range(t_maps_df.shape[0]):
    if t_maps_df.iloc[i, :-1].isna().any():
        maps_list.append(n_null)
    else: 
        # crea un array con los valores de la fila i de maps
        x_eva = t_maps_df.iloc[i, :-1]
        x_eva = x_eva.values.reshape(1, len(t_maps_df_columns))
        x_eva_df = pd.DataFrame()

        for j in range(len(t_maps_df_columns)):
            x_eva_df[t_maps_df_columns[j]] = [x_eva[j]]

        #x_eva_df = pd.DataFrame(x_eva, columns=t_maps_df_columns)
        #x_eva.columns = t_maps_df_columns
        
        pred = sklearn_lda.predict_proba(x_eva)
        maps_list.append(pred[0][1])

    load += 1
    print("progress: {0:.2f}%".format(load * 100 / max_load), end="\r")
          
'''


# crea una matriz de salida con las dimensiones del mapa de entrenamiento 
maps_df = np.zeros((t_maps_ds[0].RasterYSize, t_maps_ds[0].RasterXSize), dtype=float)

maps_df = np.reshape(maps_list, (t_maps_ds[0].RasterYSize, t_maps_ds[0].RasterXSize))

# recorre la matriz de salida
'''for i in range(maps_df.shape[0]):
    for j in range(maps_df.shape[1]):
        # reemplaza los valores de la matriz de salida por los valores de la lista de salida
        maps_df[i, j] = maps_list[i * maps_df.shape[1] + j]
'''
# convierte n_map en un mapa tematico
n_map_ds = gdal.GetDriverByName('GTiff').Create(out+'n_map_LDA.tif', t_maps_ds[0].RasterXSize, t_maps_ds[0].RasterYSize, 1, gdal.GDT_Float32)
n_map_ds.SetGeoTransform(t_maps_ds[0].GetGeoTransform())
n_map_ds.SetProjection(t_maps_ds[0].GetProjection())
n_map_ds.GetRasterBand(1).WriteArray(maps_df)
n_map_ds.GetRasterBand(1).SetNoDataValue(float(n_null))
# guarda el mapa tematico
n_map_ds = None
