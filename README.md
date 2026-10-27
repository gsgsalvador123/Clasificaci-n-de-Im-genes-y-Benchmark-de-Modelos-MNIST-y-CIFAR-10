# Benchmark de Clasificación de Imágenes: MNIST vs. CIFAR-10 con Modelos Clásicos

Este repositorio contiene un estudio comparativo y experimental de tres algoritmos clásicos de Machine Learning (**k-Nearest Neighbors**, **Regresión Logística multinomial** y **Support Vector Machines lineales**) evaluados sobre dos conjuntos de datos canónicos de visión por computador con niveles de complejidad disímiles: **MNIST** (dígitos manuscritos centrados en escala de grises) y **CIFAR-10** (objetos naturales a color con variaciones de fondo, iluminación y perspectiva).

El objetivo principal es contrastar el comportamiento de clasificadores basados en píxeles crudos, diagnosticar problemas de sesgo/varianza mediante curvas de aprendizaje, analizar matrices de confusión e interpretar espacialmente los pesos aprendidos para entender qué representa cada modelo.

---

## Estructura del Proyecto

```text
├── clasificacion_mnist_cifar10.ipynb   # Notebook principal con código, gráficos y análisis
├── requirements.txt                    # Dependencias del proyecto
├── .gitignore                          # Archivos y carpetas excluidas de Git
└── README.md                           # Documentación técnica del proyecto
```

---

## Flujo de Trabajo y Metodología

1. **Preprocesamiento y Muestreo:**
   - **MNIST:** Descarga directa vía OpenML (70,000 imágenes de 28x28). Se extrajo una submuestra estratificada del 10% (7,000 muestras) preservando la distribución de las 10 clases para un entrenamiento eficiente y balanceado. Escala de valores normalizada a $[0, 1]$ dividiendo por 255.
   - **CIFAR-10:** Carga de imágenes 32x32x3. Se aplicó reducción de dimensionalidad convirtiendo RGB a escala de grises mediante ponderación luminosa ($0.299R + 0.587G + 0.114B$), aplanando cada imagen a 1,024 dimensiones y normalizando a $[0, 1]$.

2. **Validación Cruzada:**
   - Evaluación inicial mediante 3-fold cross-validation estratificado sobre la submuestra del 10%, reportando exactitud media (*accuracy*) y desviación estándar.

3. **Optimización con GridSearchCV:**
   - **k-NN:** Búsqueda en $k \in \{1, 3, 5, 7, 9, 11, 15\}$.
   - **Regresión Logística (L2):** Exploración logarítmica de regularización $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$.
   - **SVM Lineal (LinearSVC):** Exploración de regularización $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$.
   - Diagnóstico visual de **overfitting** (ej. $k=1$ en k-NN o $C$ elevado en modelos lineales) vs. **underfitting** ($k$ muy grande o $C \le 0.001$).

4. **Análisis de Confusión y Diagnóstico de Errores:**
   - Evaluación sobre conjunto de prueba independiente (75% train / 25% test) con Regresión Logística afinada ($C=0.1$).
   - Inspección visual de dígitos erróneamente clasificados.
   - Cálculo y graficación de la matriz de confusión multiclase.

5. **Interpretabilidad de Modelos Lineales:**
   - Mapeo de la matriz de coeficientes ($10 \times 784$) a imágenes de 28x28 píxeles con paleta divergente (`RdBu_r`).
   - Identificación de plantillas positivas (evidencia a favor) y negativas (zonas huecas o trazos incompatibles).

6. **Benchmark en CIFAR-10:**
   - Replicación del pipeline sobre CIFAR-10 para evidenciar la degradación de modelos lineales y de distancia métrica directa cuando se enfrentan a imágenes naturales sin invariancia traslacional ni extracción previa de características.

---

## Resultados Principales

### 1. MNIST (Submuestra 10%, 3-Fold CV)

| Modelo | Configuración por Defecto | Configuración Optimizada (GridSearchCV) | Desv. Estándar |
|---|---|---|---|
| **k-NN** | 0.9339 | **0.9381** ($k = 3$) | $\pm 0.0039$ |
| **Regresión Logística** | 0.8956 | **0.9036** ($C = 0.1$) | $\pm 0.0028$ |
| **SVM Lineal** | 0.8753 | **0.8976** ($C = 0.01$) | $\pm 0.0027$ |

- **k-NN** obtiene el rendimiento más alto en MNIST (~93.8%) porque los dígitos manuscritos similares forman agrupaciones locales densas en el espacio de píxeles que las distancias Euclidianas aprovechan de forma natural.
- **k = 1** muestra sobreajuste severo (score de train de 1.0 vs caída en validación); el punto óptimo se alcanza en $k = 3$.

### 2. Análisis de Matriz de Confusión (Regresión Logística)
Las principales fuentes de error ocurren en pares visualmente ambiguos:
- **4 vs. 9:** Trazos verticales con aperturas superiores similares.
- **3 vs. 5:** Curvaturas inferiores idénticas.
- **7 vs. 9:** Trazo superior horizontal con bajada inclinada.
- **5 vs. 8:** Confusión en el bucle superior cuando el trazo está cerrado.

La inspección de ejemplos mal clasificados revela que la mayoría son dígitos genuinamente ambiguos o deformados incluso para un ojo humano.

### 3. Interpretabilidad Espacial de Coeficientes
- Los pesos positivos forman contornos nítidos que replican la silueta promedio del dígito correspondiente (ej. un aro para el 0, una línea vertical central para el 1).
- Los píxeles perimetrales permanecen en 0 debido a la ausencia de trazos en los bordes.
- SVM Lineal genera mapas de decisión ligeramente más suaves por la regularización más estricta ($C=0.01$), mientras que Regresión Logística define contrastes de peso más pronunciados.

### 4. CIFAR-10 (Imágenes Naturales a Escala de Grises, 3-Fold CV)

| Modelo | Accuracy (3-fold CV) |
|---|---|
| **k-NN** | 0.2218 |
| **Regresión Logística** | 0.2162 |
| **SVM Lineal** | 0.1878 |

- Los modelos clásicos sobre píxeles crudos apenas superan el azar (10% base para 10 clases), alcanzando entre 18% y 22%.
- **Conclusión técnica:** En imágenes naturales, la variación de fondo, pose, escala e iluminación destruye la correlación directa entre píxeles y etiquetas de clase. Además, eliminar la información de color reduce sustancialmente el contexto semántico. Este resultado justifica la necesidad de descriptores invariantes o arquitecturas de Deep Learning (CNNs).

---

## Cómo Ejecutar el Proyecto

### Requisitos previos
- Python 3.9 o superior.
- Jupyter Notebook / JupyterLab o Google Colab.

### Instalación del entorno
```bash
# Clonar el repositorio
git clone https://github.com/gsgsalvador123/Clasificaci-n-de-Im-genes-y-Benchmark-de-Modelos-MNIST-y-CIFAR-10.git
cd clasificacion-imagenes-mnist-cifar10

# Crear y activar entorno virtual (opcional pero recomendado)
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución
```bash
jupyter notebook clasificacion_mnist_cifar10.ipynb
```

*Nota: No se requieren descargas manuales de datos; los datasets son descargados automáticamente por las librerías `openml` y `keras.datasets` durante la ejecución.*

---

## Autor
- **Gonzalo Salvador**

