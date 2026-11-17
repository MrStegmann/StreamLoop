# Informe de Ajuste de Hiperparámetros: Modelo de Predicción de Churn StreamLoop

## 1. Resumen Ejecutivo y Métricas Comparativas

Este documento consolida los resultados del proceso de optimización sistemática y validación rigurosa del modelo de predicción de abandono de clientes (churn) de StreamLoop.

### Métricas en Conjunto de Prueba Retenido (Holdout Test Set)

| Métrica | Modelo Base (Default) | Modelo Ajustado (Ganador 1-SE) | Mejora Relativa |
| :--- | :---: | :---: | :---: |
| **F2 Score** (Métrica Primaria) | 0.6031 | **0.7209** | **+19.5%** |
| **Recall** (Sensibilidad Churn) | 0.6176 | **0.7968** | **+29.0%** |
| **Precision** | 0.5513 | 0.5219 | -5.3% |
| **F1 Score** | 0.5826 | 0.6307 | +8.3% |
| **PR-AUC** | 0.6018 | 0.6549 | +8.8% |
| **ROC-AUC** | 0.8228 | 0.8427 | +2.4% |

### Matrices de Confusión Comparativas

#### Modelo Base:
- Verdaderos Negativos (TN): 847 | Falsos Positivos (FP): 188
- Falsos Negativos (FN): 143 | Verdaderos Positivos (TP): 231

#### Modelo Ajustado (Ganador 1-SE):
- Verdaderos Negativos (TN): 762 | Falsos Positivos (FP): 273
- Falsos Negativos (FN): 76 | Verdaderos Positivos (TP): 298

### Verificación de Aislamiento y Ausencia de Fuga de Datos
Se verificó formalmente el cumplimiento del protocolo de aislamiento: el conjunto de prueba retenido fue evaluado exactamente **2 veces** (Touch 1: evaluación de referencia base; Touch 2: evaluación final del estimador ganador refactorizado). Cero transformaciones ni divisiones de validación cruzada tuvieron acceso a los datos de test, garantizando ausencia total de fuga de datos (zero data leakage).

---

## 2. Justificación de la Métrica de Negocio

### Coste Asimétrico en la Retención de Clientes
En el contexto de servicios de suscripción de StreamLoop, los errores de clasificación conllevan costes comerciales radicalmente asimétricos:
1. **Falso Negativo (FN)**: Un cliente que cancelará su suscripción no es detectado a tiempo. Esto genera la pérdida irreversible del valor de vida del cliente (Customer Lifetime Value - LTV) y exige un alto coste de adquisición para reemplazarlo.
2. **Falso Positivo (FP)**: Un cliente leal es clasificado erróneamente como propenso a cancelar. La consecuencia operativa es únicamente el envío de una oferta o descuento preventivo de bajo coste.

Dado que el coste de perder un suscriptor supera con creces el coste de una campaña preventiva, la prioridad analítica es maximizar el **Recall** (captura de casos de churn) sin degradar por completo la precisión.

### Inadecuación del Accuracy y Limitación de F1 Score
- **Accuracy**: En un conjunto con desbalance (aprox. 73.5% retenidos vs. 26.5% churn), un clasificador ingenuo que siempre prediga 'No Churn' alcanzaría un 73.5% de exactitud mientras omite el 100% de los canceladores.
- **F1 Score**: Pondera de manera simétrica la precisión y el recall (beta = 1.0), penalizando en exceso los falsos positivos y conduciendo a modelos conservadores que pasan por alto a clientes en riesgo crítico.

### Formulación Matemática de F2 Score
La métrica $F_2$ fija $\beta = 2$, otorgando al recall el doble de importancia que a la precisión:

$$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{P \cdot R}{4P + R}$$

---

## 3. Exploración de Hiperparámetros y Espacio de Búsqueda

### Estrategia de Optimización en Dos Etapas
1. **Etapa 1 (Exploración Global)**: Búsqueda aleatoria (`RandomizedSearchCV`) con 50 iteraciones y validación cruzada estratificada de 5 pliegues (250 ajustes totales), evaluando combinaciones en rangos amplios del clasificador `RandomForestClassifier`.
2. **Etapa 2 (Refinamiento Focalizado)**: Búsqueda exhaustiva en cuadrícula (`GridSearchCV`) centrada en la vecindad de la configuración más prometedora identificada en la Etapa 1.

### Espacio de Hiperparámetros y Configuración Seleccionada

| Hiperparámetro | Rango / Opciones Exploradas | Modelo Base | Configuración Ganadora (1-SE) |
| :--- | :--- | :---: | :---: |
| `n_estimators` | [50, 100, 150, 200, 250, 300, 400] | 100 | **100** |
| `max_depth` | [None, 5, 8, 10, 12, 15, 20] | None | **7** |
| `min_samples_split` | [2, 5, 10, 15, 20] | 2 | **5** |
| `min_samples_leaf` | [1, 2, 4, 6, 8, 10] | 1 | **2** |
| `max_features` | ['sqrt', 'log2', 0.2, 0.4, 0.6] | 'sqrt' | **0.2** |
| `criterion` | ['gini', 'entropy'] | 'gini' | **gini** |
| `class_weight` | ['balanced', 'balanced_subsample'] | 'balanced' | **balanced** |

---

## 4. Análisis de Estabilidad y Criterio de Selección (Regla 1-SE)

### Principio de Selección de 1 Error Estándar (1-SE Rule)
Seleccionar ciegamente la configuración con el mayor promedio de validación cruzada frecuentemente conduce a sobreajuste hacia las particiones de entrenamiento particulares. La regla de 1 Error Estándar (1-SE) formaliza el principio de parsimonia y estabilidad:

$$\text{SE} = \frac{\sigma}{\sqrt{k}} = \frac{\sigma}{\sqrt{5}}$$

Se calcula el umbral de aceptación $T = \bar{S}_{\max} - \text{SE}_{\max} = 0.7299 - 0.0066 = 0.7233$. Cualquier modelo cuya media de F2 sea igual o superior a dicho umbral se considera estadísticamente equivalente. Entre los candidatos calificados, se selecciona aquel con la menor desviación estándar entre pliegues ($\sigma$), garantizando la máxima estabilidad y generalización en producción.

### Tabla Comparativa de Candidatos Top en Validación Cruzada

| Candidato ID | Media F2 en CV ($\bar{S}$) | Desv. Estándar ($\sigma$) | Error Estándar ($\text{SE}$) | ¿Cumple 1-SE? | Estado |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #10 | 0.7299 | 0.0148 | 0.0066 | Sí | Mayor Media CV |
| #14 | 0.7294 | 0.0156 | 0.0070 | Sí | Candidato |
| #8 | 0.7289 | 0.0150 | 0.0067 | Sí | Candidato |
| #12 | 0.7282 | 0.0148 | 0.0066 | Sí | Candidato |
| #13 | 0.7277 | 0.0166 | 0.0074 | Sí | Candidato |
| #22 | 0.7237 | 0.0134 | 0.0060 | Sí | Ganador Seleccionado (1-SE) |

### Justificación del Modelo Seleccionado
El candidato **#22** fue seleccionado porque, además de situarse holgadamente dentro del umbral estadístico 1-SE (F2 medio de 0.7237 frente a umbral 0.7233), presentó la menor variabilidad entre pliegues ($\sigma = 0.0134$). Esto garantiza que el clasificador no es vulnerable a variaciones locales en la distribución de datos y ofrecerá un comportamiento predecible y robusto durante su despliegue operativo en StreamLoop.
