PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# CI gate: compile + unit tests. MLX/Metal micro-tests are fine here (same
# machine); anything needing a locally cached big model is marked "integration"
# (those tests also self-skip when the model isn't cached).
check:
	@$(PY) -m compileall -q src scripts tests
	@$(PY) -m pytest -q tests -m "not integration" -p no:cacheprovider
.PHONY: check
