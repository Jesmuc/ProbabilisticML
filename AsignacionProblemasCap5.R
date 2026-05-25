# Asigna al azar dos problemas distintos del 5.1 al 5.14 a cada persona.

personas <- c(
  "Israel",
  "Daniel",
  "Carlos",
  "Jesus",
  "Dr. Almaguer",
  "Jose Carlos",
  "Ulises"
)

problemas <- paste0("5.", 1:14)

asignar_problemas <- function(personas, problemas, problemas_por_persona = 2,
                               semilla = NULL) {
  if (!is.null(semilla)) {
    set.seed(semilla)
  }

  total_requerido <- length(personas) * problemas_por_persona

  if (total_requerido != length(problemas)) {
    stop("La cantidad de problemas no coincide con el reparto solicitado.")
  }

  problemas_mezclados <- sample(problemas)
  grupos <- split(problemas_mezclados, rep(personas, each = problemas_por_persona))

  data.frame(
    Persona = names(grupos),
    Problema_1 = sapply(grupos, `[`, 1),
    Problema_2 = sapply(grupos, `[`, 2),
    row.names = NULL,
    check.names = FALSE
  )
}

# Cambia la semilla si quieres poder repetir exactamente el mismo sorteo.
asignacion <- asignar_problemas(personas, problemas, semilla = 12345)

print(asignacion)
