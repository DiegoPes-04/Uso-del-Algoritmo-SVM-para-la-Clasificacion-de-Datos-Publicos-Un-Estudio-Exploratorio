# Uso del Algoritmo SVM para la Clasificación de Datos Públicos: Un Estudio Exploratorio.  

> **Trabajo de Grado — Diplomado en Inteligencia Artificial Procesamiento de Lenguaje Natural - PLN**  
> Universidad Santiago de Cali (USC) · Facultad de Ingeniería · Ingeniería de Sistemas

---

## 👥 Autores

| Nombre | Correo institucional |
|--------|---------------------|
| Diego Andres Pesillo Montilla | diego.pesillo00@usc.edu.co |
| Jhon Alexander Giraldo Gonzales | jhon.giraldo05@usc.edu.co |

**Directores:** Maritza Palacios Medina · Victor Viera Balanta  
**Grupo de investigación:** COMBA I+D · Universidad Santiago de Cali  
**Fecha de publicación:** Marzo 2026  
**Versión:** 1.0

---

## 📋 Descripción del proyecto

Este repositorio contiene dos notebooks de Google Colab diseñados como **guía práctica y reproducible** para aprender a implementar el algoritmo **Support Vector Machine (SVM)** aplicado al clásico dataset Iris. El proyecto nació como complemento práctico del artículo de revisión *"Uso del Algoritmo SVM para la Clasificación de Datos Públicos: Un Estudio Exploratorio"*, y está pensado para estudiantes y profesionales que se están iniciando en el campo de la inteligencia artificial y el aprendizaje automático supervisado.

La propuesta responde a una necesidad concreta: **cerrar la brecha entre la teoría y la práctica** del SVM. Para ello se documentan paso a paso las etapas de limpieza de datos, análisis exploratorio (EDA), entrenamiento del modelo, evaluación de métricas y comparación de kernels, todo desde una plataforma gratuita y sin instalación local.

### ¿Por qué dos notebooks?

Se construyeron intencionalmente **dos versiones del experimento** para ilustrar el impacto de los hiperparámetros:

| | Notebook 1 | Notebook 2 |
|--|------------|------------|
| **Hiperparámetros** | Sin ajuste (`default`) | Con `GridSearchCV` (C, gamma, degree) |
| **Mejor kernel** | Linear (~98%) | Todos superan el 90% |
| **Peor kernel** | Sigmoid (~8%) | Sigmoid >90% con ajuste |
| **Objetivo principal** | Entender el efecto del kernel | Entender el efecto de C y gamma |
| **Nivel de complejidad** | Básico-intermedio | Intermedio-avanzado |

> 💡 **Conclusión clave:** El kernel sigmoid pasa de un rendimiento muy bajo (~8%) a superar el 90% simplemente ajustando los parámetros `C` y `gamma`. Esto demuestra que **el ajuste de hiperparámetros es tan importante como la elección del kernel**.

---

##  ¿Cómo usar estos notebooks?

### Abrir directamente en Google Colab (recomendado)

Haz clic en el botón correspondiente a cada notebook:

| Notebook | Descripción | Abrir en Colab |
|----------|-------------|----------------|
| Notebook 1 — Sin hiperparámetros | EDA + SVM básico (4 kernels) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x6wETeqkVwoiXIWX7F9936JRhqoJYew4?usp=sharing) |
| Notebook 2 — Con hiperparámetros | EDA + SVM avanzado (GridSearchCV) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q7Jrq4zQmD4Ue3GfnOtDJLZGeLT3kaKj?usp=sharing) |

---

## 📂 Estructura del repositorio

```
Uso-del-Algoritmo-SVM-para-la-Clasificacion-de-Datos-Publicos-Un-Estudio-Exploratorio/
│
├── data/
│   └── Iris.csv                                # Dataset Iris (UCI Repository)
│
├── notebooks/
│   ├── SVM_DatasetIris_ConHiperparametros_Articulo.ipynb     # Notebook 1: Con GridSearchCV
│   └── SVM_DatasetIris_SinHipermarametros_Articulo.ipynb     # Notebook 2: Sin hiperparámetros
│
└── README.md
```

---

## 📓 Descripción detallada de los notebooks

### Notebook 1 — SVM básico (sin ajuste de hiperparámetros)

**Archivo:** `SVM_DatasetIris_SinHipermarametros_Articulo.ipynb`

Este notebook es el punto de entrada ideal para quienes están comenzando con SVM. Se trabaja con los parámetros por defecto de scikit-learn y se comparan los 4 kernels disponibles.

#### Paso a paso del notebook:

**Paso 1 — Carga y limpieza de datos**
- Se carga el archivo `Iris.csv` con `pandas`
- Se elimina la columna `Id` (no aporta información al modelo)
- Se codifican las especies como valores numéricos: `Iris-setosa=1`, `Iris-versicolor=2`, `Iris-virginica=3`
- Se verifica el tipo de datos con `.info()`

**Paso 2 — Análisis Exploratorio de Datos (EDA)**
- Estadísticas descriptivas con `.describe()`
- Matriz de correlación entre variables
- Pairplot coloreado por especie
- Distribución de clases (balance del dataset)
- Histogramas y boxplots de cada feature
- Visualización 3D interactiva con Plotly
- Análisis de Componentes Principales (PCA)
- Importancia de variables según PCA

**Paso 3 — Entrenamiento del modelo SVM**
- Se prueban los 4 kernels: `linear`, `rbf`, `poly`, `sigmoid`
- Validación cruzada estratificada `StratifiedKFold` con 5 particiones
- Para cada kernel se genera:
  - Precisión promedio y desviación estándar
  - Tiempo de entrenamiento
  - Matriz de confusión
  - Reporte de clasificación (precision, recall, F1-score)

**Paso 4 — Evaluación y visualización final**
- Gráfico comparativo de precisión por kernel (con barras de error)
- Gráfico comparativo de tiempos de entrenamiento
- Curvas ROC multiclase (AUC por clase) para cada kernel
- Predicción de ejemplo con nueva muestra

#### Resultados esperados (Notebook 1):

| Kernel | Precisión aprox. |
|--------|-----------------|
| Linear | ~98% ✅ |
| RBF | ~96% |
| Poly | ~98% |
| Sigmoid | ~8% ❌ |

---

### Notebook 2 — SVM avanzado (con GridSearchCV e hiperparámetros)

**Archivo:** `SVM_DatasetIris_ConHiperparametros_Articulo.ipynb`

Este notebook es la versión extendida. Introduce buenas prácticas como el uso de `Pipeline`, `StandardScaler` y búsqueda de hiperparámetros con `GridSearchCV`. También incluye un modelo base (baseline) de comparación con Regresión Logística.

#### Paso a paso del notebook:

**Fase 1 — Selección de datos** *(O - Obtain)*
- Carga del dataset desde `Iris.csv`
- Verificación de dimensiones y columnas

**Fase 2 — Preprocesamiento y limpieza** *(S - Scrub)*
- Eliminación de columna `Id`
- Verificación de valores nulos
- Balance de clases con visualización

**Fase 3 — Transformación y EDA** *(E - Explore)*
- Estadísticas descriptivas completas
- Matriz de correlación con heatmap
- Pairplot con seaborn
- Histogramas y boxplots por feature
- Escalado estándar (`StandardScaler`)
- División estratificada: 70% entrenamiento / 30% prueba

**Fase 4 — Modelado** *(M - Model)*
- **Baseline:** Regresión Logística (para comparación)
- **GridSearchCV** para cada kernel con los siguientes rangos de búsqueda:

  | Kernel | Parámetros buscados |
  |--------|---------------------|
  | Linear | C: [0.1, 1, 10, 100] |
  | RBF | C: [0.1, 1, 10, 100], gamma: ['scale', 'auto', 0.001, 0.01, 0.1] |
  | Poly | C: [0.1, 1, 10, 100], gamma: ['scale', 'auto', 0.001, 0.01, 0.1], degree: [2, 3, 4] |
  | Sigmoid | C: [0.1, 1, 10, 100], gamma: ['scale', 'auto', 0.001, 0.01, 0.1] |

- Se reportan los mejores hiperparámetros encontrados para cada kernel
- Métricas: accuracy, precision, recall, F1-score, AUC-ROC

**Fase 5 — Interpretación** *(N - iNterpret)*
- Tabla comparativa de todos los kernels optimizados
- Curvas de validación (para visualizar sesgo/varianza)
- Curvas de aprendizaje
- Matrices de confusión
- Curvas ROC multiclase

#### Resultados esperados (Notebook 2):

| Kernel | Precisión aprox. (con ajuste) |
|--------|-------------------------------|
| Linear | ~94% ✅ |
| RBF | ~91% ✅ |
| Poly | ~92% ✅ |
| Sigmoid | ~91% ✅ |

> 🔍 Comparar estos resultados con el Notebook 1 es el principal aporte didáctico del proyecto.

---

## 🗂️ Dataset Iris

| Característica | Detalle |
|---------------|---------|
| **Origen** | UCI Machine Learning Repository |
| **Autor** | R.A. Fisher (1936) |
| **Muestras** | 150 (50 por clase) |
| **Variables** | 4 features + 1 target |
| **Clases** | Iris setosa, Iris versicolor, Iris virginica |
| **Valores nulos** | Ninguno |
| **Acceso** | https://doi.org/10.24432/C56C76 |

### Variables del dataset:

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `SepalLengthCm` | Longitud del sépalo (cm) | Numérica |
| `SepalWidthCm` | Ancho del sépalo (cm) | Numérica |
| `PetalLengthCm` | Longitud del pétalo (cm) | Numérica |
| `PetalWidthCm` | Ancho del pétalo (cm) | Numérica |
| `Species` | Especie de la flor | Categórica (target) |

---

## 🛠️ Librerías utilizadas

```python
numpy          # Operaciones numéricas
pandas         # Manipulación de datos
matplotlib     # Visualizaciones estáticas
seaborn        # Visualizaciones estadísticas
scikit-learn   # Modelos SVM, métricas, validación cruzada, GridSearchCV
mlxtend        # Visualización mejorada de matrices de confusión (Notebook 1)
plotly         # Visualización 3D interactiva (Notebook 1)
```

> Todas las librerías están **preinstaladas en Google Colab**. No es necesario instalar nada manualmente.

---

## ⚙️ Conceptos clave explicados en los notebooks

### ¿Qué es SVM?
El Support Vector Machine (SVM) es un algoritmo de aprendizaje supervisado que busca el **hiperplano óptimo** que maximiza el margen entre clases. Los puntos más cercanos a ese límite se llaman **vectores de soporte**.

### ¿Qué es un kernel?
Un kernel es una función matemática que transforma los datos a un espacio de mayor dimensión para hacer separables clases que no lo son en el espacio original.

| Kernel | Cuándo usarlo |
|--------|---------------|
| `linear` | Datos linealmente separables |
| `rbf` (Gaussiano) | Casos generales, el más usado |
| `poly` | Relaciones polinomiales entre variables |
| `sigmoid` | Redes neuronales superficiales; requiere ajuste fino |

### ¿Qué son C y gamma?
- **C (regularización):** Controla el equilibrio entre maximizar el margen y minimizar errores de clasificación. Un C grande = menos tolerancia al error (riesgo de overfitting).
- **gamma:** Define qué tan lejos "llega" la influencia de un punto de entrenamiento. Solo aplica a kernels RBF, poly y sigmoid.

### ¿Por qué usar GridSearchCV?
`GridSearchCV` prueba sistemáticamente todas las combinaciones posibles de hiperparámetros y selecciona la que produce mejor rendimiento mediante **validación cruzada**. Esto elimina el ensayo y error manual.

### ¿Qué es la validación cruzada estratificada (StratifiedKFold)?
Divide el dataset en `k` particiones manteniendo la misma proporción de clases en cada partición. Así las métricas son más confiables y robustas.

---

## Metodología aplicada (OSEMN)

Los notebooks siguen la metodología **OSEMN** para estructurar el flujo de trabajo:

```
O → Obtain   (Obtener el dataset Iris desde repositorio público)
S → Scrub    (Limpiar: eliminar Id, verificar nulos, verificar tipos)
E → Explore  (EDA: correlaciones, distribuciones, pairplots, PCA)
M → Model    (Entrenar SVM con distintos kernels e hiperparámetros)
N → iNterpret (Comparar métricas, matrices de confusión, curvas ROC)
```

---

## 📖 Cómo citar este repositorio

Si usas este repositorio, sus gráficos o su código en tu trabajo académico, cita de la siguiente manera:

**Formato APA:**

> Pesillo Montilla, D. A., & Giraldo Gonzales, J. A. (2026). *Uso del Algoritmo SVM para la Clasificación de Datos Públicos: Un Estudio Exploratorio* [Repositorio de GitHub]. Universidad Santiago de Cali. https://github.com/DiegoPes-04/Uso-del-Algoritmo-SVM-para-la-Clasificacion-de-Datos-Publicos-Un-Estudio-Exploratorio

**Formato BibTeX:**

```bibtex
@misc{pesillo2026svm,
  author       = {Pesillo Montilla, Diego Andres and Giraldo Gonzales, Jhon Alexander},
  title        = {Uso del Algoritmo SVM para la Clasificación de Datos Públicos: Un Estudio Exploratorio},
  year         = {2026},
  institution  = {Universidad Santiago de Cali},
  howpublished = {\url{https://github.com/DiegoPes-04/Uso-del-Algoritmo-SVM-para-la-Clasificacion-de-Datos-Publicos-Un-Estudio-Exploratorio}},
  note         = {Trabajo de Grado – Diplomado en Inteligencia Artificial Procesamiento de Lenguaje Natural - PLN}
}
```

---

## 📚 Referencias principales

- Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273–297.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
- Cervantes, J. et al. (2020). A comprehensive survey on support vector machine classification. *Neurocomputing*, 408, 301–321.

---

## 📝 Licencia

Este proyecto fue desarrollado con fines académicos y educativos en el marco del Diplomado en Inteligencia Artificial Procesamiento de Lenguaje Natural - PLN de la Universidad Santiago de Cali. El dataset Iris es de dominio público (Fisher, 1936).

---

*Hecho con ❤️ en Cali, Colombia 🇨🇴*
