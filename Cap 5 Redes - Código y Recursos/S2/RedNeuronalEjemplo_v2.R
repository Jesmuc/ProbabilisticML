# ============================================================
# Red neuronal simple para regresión 1D en R
# ------------------------------------------------------------
# Este script acompaña las diapositivas de la Sesión 2.
# El objetivo es mostrar, de forma didáctica, cómo se implementan
# las ecuaciones de propagación hacia adelante y retropropagación.
#
# Arquitectura usada en el ejemplo:
#   - 1 entrada
#   - 1 capa oculta con H = 3 neuronas y activación tanh
#   - 1 salida lineal
#
# Notación de parámetros:
#   - W1: pesos de la entrada a la capa oculta, W^(1)
#   - b1: sesgos de la capa oculta, b^(1)
#   - W2: pesos de la capa oculta a la salida, W^(2)
#   - b2: sesgo de la salida, b^(2)
#
# Notación de variables intermedias:
#   - a1: preactivación de la capa oculta, a^(1) = W1 x + b1
#   - z1: activación de la capa oculta, z^(1) = tanh(a1)
#   - a2: preactivación de salida, a^(2) = W2 z1 + b2
#   - y_hat: predicción final. Como la salida es lineal, y_hat = a2
#
# En la Sesión 2 usamos E_n = 1/2 (y_hat - y)^2 para un solo dato.
# Por eso la señal de salida es delta2 = y_hat - y.
# ============================================================

set.seed(42)

# ------------------------------------------------------------
# 1. Función objetivo y datos de entrenamiento
# ------------------------------------------------------------

# Función que queremos aproximar con la red.
funcion_objetivo <- function(x) {
  sin(x)
}

# Generamos datos en el intervalo [-pi, pi].
# x_entrenamiento: entradas
# y_entrenamiento: salidas deseadas
x_entrenamiento <- seq(-pi, pi, length.out = 60)
y_entrenamiento <- funcion_objetivo(x_entrenamiento)

# ------------------------------------------------------------
# 2. Inicialización de parámetros
# ------------------------------------------------------------

# input_size : número de variables de entrada. En este ejemplo, 1.
# hidden_size: número de neuronas ocultas. En las diapositivas fijamos H = 3.
# output_size: número de salidas. En este ejemplo, 1.
inicializar_parametros <- function(input_size, hidden_size, output_size) {
  list(
    W1 = matrix(
      rnorm(hidden_size * input_size, sd = 0.5),
      nrow = hidden_size,
      ncol = input_size
    ),
    b1 = matrix(0, nrow = hidden_size, ncol = 1),
    W2 = matrix(
      rnorm(output_size * hidden_size, sd = 0.5),
      nrow = output_size,
      ncol = hidden_size
    ),
    b2 = matrix(0, nrow = output_size, ncol = 1)
  )
}

# ------------------------------------------------------------
# 3. Funciones auxiliares
# ------------------------------------------------------------

# Derivada de tanh expresada en términos de la salida activada z.
# Si z = tanh(a), entonces d/da tanh(a) = 1 - tanh(a)^2 = 1 - z^2.
derivada_tanh_desde_z <- function(z) {
  1 - z^2
}

# Error cuadrático medio para monitorear el entrenamiento completo.
# Este ECM no es la pérdida de un solo dato; solo se usa para visualizar progreso.
error_cuadratico_medio <- function(valor_real, valor_predicho) {
  mean((valor_real - valor_predicho)^2)
}

# Pérdida para un solo dato, consistente con la notación de la Sesión 2:
#   E_n = 1/2 (y_hat - y)^2
perdida_un_dato <- function(y_real, y_hat) {
  0.5 * (y_hat - y_real)^2
}

# ------------------------------------------------------------
# 4. Propagación hacia adelante
# ------------------------------------------------------------

# x      : una sola entrada, almacenada como matriz columna 1x1
# params : lista con W1, b1, W2, b2
#
# Ecuaciones:
#   a1    = W1 %*% x + b1
#   z1    = tanh(a1)
#   a2    = W2 %*% z1 + b2
#   y_hat = a2    porque la salida es lineal
propagacion_adelante <- function(x, params) {
  a1 <- params$W1 %*% x + params$b1
  z1 <- tanh(a1)

  a2 <- params$W2 %*% z1 + params$b2
  y_hat <- a2

  list(
    x = x,
    a1 = a1,
    z1 = z1,
    a2 = a2,
    y_hat = y_hat
  )
}

# ------------------------------------------------------------
# 5. Retropropagación
# ------------------------------------------------------------

# Calcula los gradientes de E_n respecto a todos los parámetros
# para un solo dato de entrenamiento.
#
# Fórmulas usadas:
#   delta2 = dE_n/da2 = y_hat - y
#   dW2    = delta2 %*% t(z1)
#   db2    = delta2
#   delta1 = (t(W2) %*% delta2) * (1 - z1^2)
#   dW1    = delta1 %*% t(x)
#   db1    = delta1
retropropagacion <- function(cache, y_real, params) {
  # ---- Capa de salida ----
  # Como E_n = 1/2 (y_hat - y)^2 y y_hat = a2,
  # se obtiene delta2 = dE_n/da2 = y_hat - y.
  delta2 <- cache$y_hat - y_real

  dW2 <- delta2 %*% t(cache$z1)
  db2 <- delta2

  # ---- Capa oculta ----
  # Primero propagamos la sensibilidad hacia z1 mediante W2^T.
  # Después multiplicamos elemento a elemento por la derivada de tanh.
  delta1 <- (t(params$W2) %*% delta2) * derivada_tanh_desde_z(cache$z1)

  dW1 <- delta1 %*% t(cache$x)
  db1 <- delta1

  list(
    dW1 = dW1,
    db1 = db1,
    dW2 = dW2,
    db2 = db2,
    delta1 = delta1,
    delta2 = delta2
  )
}

# ------------------------------------------------------------
# 6. Actualización por descenso por gradiente estocástico
# ------------------------------------------------------------

# lr: tasa de aprendizaje; controla el tamaño de cada paso.
actualizar_parametros_sgd <- function(params, grads, lr) {
  params$W1 <- params$W1 - lr * grads$dW1
  params$b1 <- params$b1 - lr * grads$db1
  params$W2 <- params$W2 - lr * grads$dW2
  params$b2 <- params$b2 - lr * grads$db2
  params
}

# ------------------------------------------------------------
# 7. Predicción para varios puntos
# ------------------------------------------------------------

# Ejecuta la red para un vector completo de entradas.
prediccion_red <- function(x_valores, params) {
  sapply(x_valores, function(x) {
    x_matriz <- matrix(x, nrow = 1, ncol = 1)
    propagacion_adelante(x_matriz, params)$y_hat[1, 1]
  })
}

# ------------------------------------------------------------
# 8. Entrenamiento de la red
# ------------------------------------------------------------

# x_valores      : vector con entradas de entrenamiento
# y_valores      : vector con salidas deseadas
# hidden_size    : número de neuronas ocultas. Usamos 3 por claridad didáctica.
# lr             : tasa de aprendizaje
# max_steps      : número máximo de actualizaciones SGD
# error_threshold: criterio simple de parada temprana medido con ECM
entrenar_red <- function(x_valores,
                         y_valores,
                         hidden_size = 3,
                         lr = 0.02,
                         max_steps = 30000,
                         error_threshold = 0.001) {

  params <- inicializar_parametros(
    input_size = 1,
    hidden_size = hidden_size,
    output_size = 1
  )

  historial_ecm <- numeric(max_steps)

  for (step in 1:max_steps) {
    # Elegimos un solo dato aleatorio para hacer una actualización SGD.
    i <- sample.int(length(x_valores), size = 1)

    x <- matrix(x_valores[i], nrow = 1, ncol = 1)
    y <- matrix(y_valores[i], nrow = 1, ncol = 1)

    cache <- propagacion_adelante(x, params)
    grads <- retropropagacion(cache, y, params)
    params <- actualizar_parametros_sgd(params, grads, lr)

    # Monitoreamos el error sobre todo el conjunto de entrenamiento.
    predicciones <- prediccion_red(x_valores, params)
    historial_ecm[step] <- error_cuadratico_medio(y_valores, predicciones)

    # Parada temprana sencilla.
    if (historial_ecm[step] < error_threshold) {
      historial_ecm <- historial_ecm[1:step]
      cat("Entrenamiento detenido en el paso", step,
          "con ECM =", historial_ecm[step], "\n")
      break
    }
  }

  list(
    params = params,
    historial_ecm = historial_ecm
  )
}

# ------------------------------------------------------------
# 9. Ejecutar entrenamiento
# ------------------------------------------------------------

modelo <- entrenar_red(
  x_valores = x_entrenamiento,
  y_valores = y_entrenamiento,
  hidden_size = 3,
  lr = 0.025,
  max_steps = 300000,
  error_threshold = 0.0001
)

# Predicciones finales sobre los puntos de entrenamiento.
y_predicho <- prediccion_red(x_entrenamiento, modelo$params)

# Error final sobre todo el conjunto.
error_final <- error_cuadratico_medio(y_entrenamiento, y_predicho)
cat("ECM final:", error_final, "\n")

# ------------------------------------------------------------
# 10. Visualización de resultados
# ------------------------------------------------------------

# Gráfica de la función objetivo y de la aproximación de la red.
plot(x_entrenamiento, y_entrenamiento,
     pch = 16,
     col = "steelblue",
     main = "Regresión con una red neuronal simple",
     xlab = "x",
     ylab = "y")
lines(x_entrenamiento, y_predicho, col = "firebrick", lwd = 2)
legend("topright",
       inset = 0.02,
       legend = c("Datos objetivo", "Predicción de la red"),
       col = c("steelblue", "firebrick"),
       pch = c(16, NA),
       lty = c(NA, 1),
       lwd = c(NA, 2),
       bty = "n",
       cex = 0.9,
       pt.cex = 0.9,
       x.intersp = 0.6,
       y.intersp = 0.8,
       seg.len = 1.5)

# Gráfica del error durante el entrenamiento.
plot(modelo$historial_ecm,
     type = "l",
     lwd = 2,
     col = "darkgreen",
     main = "Error durante el entrenamiento",
     xlab = "Paso",
     ylab = "ECM")
