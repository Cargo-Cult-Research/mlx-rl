PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# CI gate: compile + unit tests. MLX/Metal micro-tests are fine here
# (same machine); anything needing a big model or a live server goes under -m live.
check:
	@$(PY) -m compileall -q src scripts tests
	@$(PY) -m pytest -q tests -m "not live" -p no:cacheprovider
.PHONY: check
