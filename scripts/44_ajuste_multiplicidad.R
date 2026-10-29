#!/usr/bin/env Rscript
# R.6: Holm-Bonferroni y Benjamini-Hochberg sobre las pruebas confirmatorias.
#
# Entrada : entregables/fuentes/tablas/tabla10_significancia_pareada.csv
# Salidas : entregables/fuentes/tablas/tabla10_significancia_pareada_ajustada.csv
#           entregables/fuentes/tablas/tablaS07_significancia_ajustada.md
#           Tabla S7 insertada entre marcadores en los dos suplementos.
#
# Uso: Rscript scripts/44_ajuste_multiplicidad.R [raiz_del_repositorio]

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1L) normalizePath(args[[1]]) else normalizePath(".")
dir_tab <- file.path(root, "entregables", "fuentes", "tablas")

entrada    <- file.path(dir_tab, "tabla10_significancia_pareada.csv")
salida_csv <- file.path(dir_tab, "tabla10_significancia_pareada_ajustada.csv")
salida_md  <- file.path(dir_tab, "tablaS07_significancia_ajustada.md")

sig <- read.csv(entrada, stringsAsFactors = FALSE, check.names = FALSE,
                fileEncoding = "UTF-8")

req <- c("region", "task", "comparacion", "delta_roc",
         "p_value_bootstrap", "n_bootstrap")
faltan <- setdiff(req, names(sig))
if (length(faltan)) stop("Faltan columnas: ", paste(faltan, collapse = ", "))
if (nrow(sig) == 0L) stop("La tabla de significancia esta vacia")
if (any(!is.finite(sig$p_value_bootstrap))) stop("p-values no finitos")
if (any(sig$p_value_bootstrap <= 0 | sig$p_value_bootstrap > 1))
  stop("p-values fuera de (0, 1]")

m <- nrow(sig)
p <- sig$p_value_bootstrap

sig$familia <- ifelse(grepl("vs Canadian FWI|vs INPE", sig$comparacion),
                      "benchmark_externo", "ablacion_interna")
sig$p_holm      <- p.adjust(p, method = "holm")
sig$p_bh        <- p.adjust(p, method = "BH")
sig$piso        <- p <= (1 / sig$n_bootstrap) + 1e-12
sig$sig_holm_05 <- sig$p_holm < 0.05
sig$sig_bh_05   <- sig$p_bh < 0.05
sig$m_familia   <- m

stopifnot(
  all(sig$p_holm >= p - 1e-9),
  all(sig$p_bh >= p - 1e-9),
  all(sig$p_bh <= sig$p_holm + 1e-9)
)

write.csv(sig, salida_csv, row.names = FALSE, quote = FALSE, fileEncoding = "UTF-8")

fmt_raw <- function(x, piso) ifelse(piso, "\u22640.001", sprintf("%.4f", x))
fmt_adj <- function(x) sprintf("%.4f", x)

enc <- c("| Region | Task | Comparison | \u0394ROC | p raw | p Holm | p BH | B |",
         "|---|---|---|---|---|---|---|---|")
fil <- sprintf("| %s | %s | %s | %.4f | %s | %s | %s | %d |",
               sig$region, sig$task, sig$comparacion, sig$delta_roc,
               fmt_raw(p, sig$piso), fmt_adj(sig$p_holm), fmt_adj(sig$p_bh),
               as.integer(sig$n_bootstrap))
writeLines(enc2utf8(c(enc, fil)), salida_md, useBytes = TRUE)

splice <- function(path, bloque) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  i <- grep("<!-- tablaS7:inicio -->", txt, fixed = TRUE)
  j <- grep("<!-- tablaS7:fin -->", txt, fixed = TRUE)
  if (length(i) != 1L || length(j) != 1L || i >= j)
    stop("Marcadores S7 invalidos en ", path)
  writeLines(enc2utf8(c(txt[seq_len(i)], "", bloque, "", txt[j:length(txt)])),
             path, useBytes = TRUE)
}

qmds <- c(file.path(root, "manuscrito", "en", "material_suplementario_en.qmd"),
          file.path(root, "manuscrito", "es", "material_suplementario.qmd"))
for (q in qmds) {
  if (file.exists(q)) splice(q, c(enc, fil))
}

cat(sprintf("m=%d | Holm max=%.4f | BH max=%.4f | todas Holm<0.05: %s\n",
            m, max(sig$p_holm), max(sig$p_bh), all(sig$sig_holm_05)))
