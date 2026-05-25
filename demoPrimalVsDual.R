#####################################
############ DATOS ##################
#####################################
# La regresion ridge puede resolverse en forma primal (pesos w)
# o en forma dual (coeficientes a sobre los datos). Ambas formas
# producen exactamente las mismas predicciones, salvo error numerico.

set.seed(42)

N <- 10
grado <- 12
lambda <- 1e-3

f_real <- function(x) sin(2 * pi * x)

x <- seq(0, 1, length.out = N)
t_obs <- f_real(x) + rnorm(N, mean = 0, sd = 0.15)

# Vector base polinomial phi(x) = [1, x, x^2, ..., x^grado]^T
phi <- function(x, grado) {
  x^(0:grado)
}

# Matriz de diseño Phi de tamaño N x M, con Phi[n, ] = phi(x_n)^T
Phi <- t(sapply(x, phi, grado = grado))
M <- ncol(Phi)

x_eval <- seq(0, 1, length.out = 400)
Phi_eval <- t(sapply(x_eval, phi, grado = grado))

#####################################
############ FORMA PRIMAL ###########
#####################################

# w = (Phi^T Phi + lambda I_M)^(-1) Phi^T t
w_primal <- solve(
  crossprod(Phi) + lambda * diag(M),
  crossprod(Phi, t_obs)
)

y_primal <- as.vector(Phi_eval %*% w_primal)
y_train_primal <- as.vector(Phi %*% w_primal)

#####################################
############ FORMA DUAL #############
#####################################

# K = Phi Phi^T es el kernel lineal inducido por la base phi.
# a = (K + lambda I_N)^(-1) t
K <- tcrossprod(Phi)

a_dual <- solve(K + lambda * diag(N), t_obs)

# Para puntos nuevos: k(x_*, X) = phi(x_*)^T Phi^T
K_eval <- Phi_eval %*% t(Phi)

y_dual <- as.vector(K_eval %*% a_dual)
y_train_dual <- as.vector(K %*% a_dual)

# La relacion entre ambas soluciones es w = Phi^T a.
w_desde_dual <- as.vector(crossprod(Phi, a_dual))

#####################################
############ VERIFICACION ###########
#####################################

dif_eval <- max(abs(y_primal - y_dual))
dif_train <- max(abs(y_train_primal - y_train_dual))
dif_w <- max(abs(as.vector(w_primal) - w_desde_dual))

cat("Max |y_primal - y_dual| en puntos de evaluacion: ",
    format(dif_eval, scientific = TRUE), "\n")
cat("Max |Phi w - K a| en entrenamiento:       ",
    format(dif_train, scientific = TRUE), "\n")
cat("Max |w_primal - Phi^T a_dual|:            ",
    format(dif_w, scientific = TRUE), "\n")
cat("Predicciones equivalentes en puntos de evaluacion?: ",
    isTRUE(all.equal(y_primal, y_dual, tolerance = 1e-8)), "\n")

#####################################
############ GRAFICOS ###############
#####################################

old_par <- par(no.readonly = TRUE)

# Usamos graficos separados para evitar el error
# "figure margins too large" cuando el panel de plots es pequeno.
par(mfrow = c(1, 1), mar = c(4, 4, 3, 1))

plot(x, t_obs,
     main = "Regresion ridge: forma primal vs dual",
     xlab = "x", ylab = "t",
     pch = 19, col = "black")
curve(f_real(x), from = 0, to = 1, add = TRUE, col = "darkgreen", lwd = 2)
lines(x_eval, y_primal, col = "blue", lwd = 2)
lines(x_eval, y_dual, col = "red", lwd = 2, lty = 2)
legend("topright",
       legend = c("Datos", "Funcion real", "Primal", "Dual"),
       col = c("black", "darkgreen", "blue", "red"),
       pch = c(19, NA, NA, NA),
       lty = c(NA, 1, 1, 2),
       lwd = c(NA, 2, 2, 2),
       bty = "n")

if (interactive()) {
  readline("Presiona ENTER para ver la diferencia numerica...")
}

plot(x_eval, y_primal - y_dual,
     type = "l", lwd = 2, col = "purple",
     main = "Diferencia numerica entre predicciones",
     xlab = "x", ylab = "y_primal - y_dual")
abline(h = 0, col = "gray40", lty = 2)

par(old_par)
