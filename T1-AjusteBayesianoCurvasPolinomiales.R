#####################################
############ DATOS ##################
#####################################
#Generar datos sintéticos
# 10 puntos de t = sin(2*pi*x) + N(0, 0.1)
set.seed(42)
grado <- 9
n <- 10
x <- seq(0, 1, length.out = n)
t <- sin(2 * pi * x) + rnorm(n, mean = 0, sd = 0.25)

#Graficar los datos sinteticos
plot(x, t, main = "Ajuste Polinomial", xlab = "x", ylab = "t", pch = 19,
     xlim = c(0, 1.2))

#Graficar curva teorica t = sin(2*pi*x)
curve(sin(2 * pi * x), add = TRUE, col = "blue", lwd = 2)

#Para obtener los pesos del ajuste polinomial, al minimizar el error cuadratico
# el vector solucion de maxima verosimilitud en un ajuste polinomial de orden M
# tiene la forma w_ML = X^(-1) * R

#####################################
############ AJUSTE PUNTUAL #########
#####################################
#Calcular el n-ésimo momento de los datos de entrada x
moment_x <- function(x, n) {
  mean(x^n)
}

#Calcular el n-ésimo momento de los datos de entrada multiplicados por t_i
# 1/N Σ (x_i^n * t_i)
moment_xt <- function(datos, n) {
  mean((datos^n)*t)
}

#Crear la matriz de momentos M+1 x M+1 para calcular los pesos del ajuste polinomial (X)
M <- matrix(0, nrow = grado + 1, ncol = grado + 1)
#Crear vector que contiene los momentos de x_i^n * t_i (R)
b <- numeric(grado + 1)

#Guardar los momentos de 0 a 2M de los datos de entrada x
moments <- sapply(0:(grado * 2), function(n) moment_x(x, n))
#Llenar matriz simétrica con los momentos guardados
for (i in 0:grado) {
  for (j in 0:grado) {
    M[i + 1, j + 1] <- moments[i + j + 1]
  }
  #Calcular vector R
  b[i + 1] <- moment_xt(x, i)
}

#Calcular pesos w = X^-1 * R
w <- solve(M, b)

#funcion para predecir t dado un x usando los pesos w
t_predict <- function(x)
{
  sum(w * x^(0:(length(w) - 1)))
}

# Vector base polinomial phi(x) = [1, x, x^2, ..., x^M]^T
phi <- function(x, grado) {
  x^(0:grado)
}

#####################################
############ GRAFICOS ################
#####################################

#Graficar la curva polinomial usando los pesos w ajustados
x_grid <- seq(0, 2, length.out = 400)           # grid x denso
p <- sapply(x_grid, function(x) t_predict(x))
lines(x_grid, p, type = "l", lwd = 2)

# Graficar el valor teórico para x_new usando t = sin(2*pi*x)
x_new <- 0.45
abline(h = sin(2 * pi * x_new), col = "red", lwd = 2)
# Graficar predicción del valor promedio para x_new
abline(h = t_predict(x_new), col = "blue", lwd = 2)

#########################################################################################################
############ CALCULAR DISTRIBUCION PREDICTIVA PARA UN NUEVO x0 (Bayes pt.1) usando estimadores ML #######
#########################################################################################################

residuos_ml <- t - sapply(x, t_predict)
sigma2_ML <- mean(residuos_ml^2)
beta_ML <- 1 / sigma2_ML

#####################################
############ GRAFICOS ################
#####################################
sd<- sqrt(1/beta_ML)

# graficar distribución normal centrada en t_predict(x_new) con varianza beta_ML^-1
# corresponde a p(t_new | x_new, w_ML, beta_ML) = N(t_new | t_predict(x_new, w_ML), beta_ML^-1)
curve(dnorm(x, mean = t_predict(x_new), sd = sd),
      from = -3*sd + t_predict(x_new), to = 3*sd + t_predict(x_new),
      lwd = 2,
      xlab = "x", ylab = "Density",
      main = "Distribución predictiva para un nuevo x_new")

#plot theoretical value for x_new using t = sin(2*pi*x)
abline(v = sin(2 * pi * x_new), col = "red", lwd = 2)

#plot mean predicted value for x_new
abline(v = t_predict(x_new), col = "blue", lwd = 2)

legend("topright",
       legend = c("Theoretical Value", "Predicted Mean"),
       col = c("red", "blue"),
       lwd = 2)

#############################################################################################
############ CALCULAR DISTRIBUCION PREDICTIVA PARA UN NUEVO x0 (Bayes pt.2) en general #######
#############################################################################################
# Considerando alfa y beta fijos (hiperparametros),
# Corresponde a eq.1: 
#               p(t_new | x_new, x, t) = integral(p(t_new | x_new, w) * p(w | x, t) dw)
# donde p(t_new | x_new, w) = N(t_new | t_predict(x_new, w), beta^-1)
#     y p(w | x, t) = p(t|x, w, beta) * p(w|alfa) / p(t|x, alfa, beta)
#     es la distribución posterior de los pesos w dados los datas (x, t)

alfa = 5*(10^-3)
beta = 11.1

#Al resolver la eq.1 se obtiene
# p(t_new | x_new, x, t) = N(t_new | m(x), s^2(x))

# Matriz de diseño Phi de tamaño N x (M + 1), con Phi[n, ] = phi(x_n)^T
Phi <- t(sapply(x, phi, grado = grado))

# S_N^{-1} = alfa*I + beta*Phi^T*Phi
S_inv <- alfa * diag(grado + 1) + beta * crossprod(Phi)
S <- solve(S_inv)

# m_N = beta * S_N * Phi^T * t
m_N <- beta * S %*% crossprod(Phi, t)

# Media predictiva m(x) = m_N^T * phi(x)
m <- function(x) {
  phi_x <- phi(x, grado)
  as.numeric(t(m_N) %*% phi_x)
}

# Varianza predictiva s^2(x) = 1/beta + phi(x)^T * S_N * phi(x)
s2 <- function(x) {
  phi_x <- phi(x, grado)
  as.numeric((1 / beta) + t(phi_x) %*% S %*% phi_x)
}

#finalmente la distribucion predictiva para un nuevo x_new es p(t_new | x_new, x, t) = N(t_new | m(x_new), s^2(x_new))
#por lo que graficamos una curva m(x) y un intervalo de confianza m(x) +/- 2*s(x) sobre los datos originales

#Graficar los datos sinteticos
plot(x, t, main = "Ajuste Polinomial", xlab = "x", ylab = "t", pch = 19,
     xlim = c(0, 1.2))

# Graficar curva teorica t = sin(2*pi*x)
curve(sin(2 * pi * x), add = TRUE, col = "darkgreen", lwd = 2)

#graficar m(x)
x_grid <- seq(0, 1.2, length.out = 400)
m_values <- sapply(x_grid, m)
lines(x_grid, m_values, col = "blue", lwd = 2)

#graficar intervalo de confianza m(x) +/- 2*s(x)
s_values <- sapply(x_grid, function(x) sqrt(s2(x)))
lines(x_grid, m_values + 2 * s_values, col = "red", lwd = 1, lty = 2)
lines(x_grid, m_values - 2 * s_values, col = "red", lwd = 1, lty = 2)

legend("topright",
       legend = c("Valor teorico", "Media predictiva bayesiana", "Banda m(x) +/- 2s(x)"),
       col = c("darkgreen", "blue", "red"),
       lwd = c(2, 2, 1),
       lty = c(1, 1, 2))

