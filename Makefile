PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
PYTEST ?= $(PYTHON) -m pytest
RSCRIPT ?= Rscript

.PHONY: help test data eval render maestros submission verify normalize clean

help:
	@printf '%s\n' \
	  'Usage: make <target>' \
	  '' \
	  '  test        Run the test suite (pytest)' \
	  '  data        Download, validate and build data and base models' \
	  '  eval        Benchmarks, sensitivity and robustness' \
	  '  render      Render manuscript and supplement (DOCX/HTML)' \
	  '  maestros    Build figuras_maestro.pdf and tablas_maestro.xlsx' \
	  '  submission  Package and verify dist/submission/' \
	  '  verify      Verify checksums, hashes and submission package' \
	  '  normalize   Keep only PNG figures and CSV tables' \
	  '  clean       Remove local caches'

test:
	$(PYTEST) -q tests

data:
	bash scripts/run_all.sh data

eval:
	bash scripts/run_all.sh benchmark

render:
	bash scripts/render_manuscrito.sh
	$(PYTHON) scripts/40_formatear_docx_cageo.py entregables/en/manuscrito_en.docx entregables/en/manuscrito_en.docx
	$(PYTHON) scripts/41_formatear_suplementario_docx.py
	$(PYTHON) scripts/59_generar_hashes.py

maestros:
	$(PYTHON) scripts/69_generar_maestros.py

submission:
	$(PYTHON) scripts/72_preparar_submission.py
	$(PYTHON) scripts/70_verificar_envio.py

verify:
	$(PYTHON) scripts/32_verificar_candidato.py
	$(PYTHON) scripts/59_generar_hashes.py
	$(PYTHON) scripts/70_verificar_envio.py

normalize:
	$(PYTHON) scripts/99_normalizar_formatos.py

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -name '.DS_Store' -delete
