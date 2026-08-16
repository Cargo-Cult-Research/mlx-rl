PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# CI gate: compile + unit tests. MLX/Metal micro-tests are fine here (same
# machine); anything needing a locally cached big model is marked "integration"
# (those tests also self-skip when the model isn't cached).
check:
	@$(PY) -m compileall -q src scripts tests
	@$(PY) -m pytest -q tests -m "not integration" -p no:cacheprovider
.PHONY: check

# Reapply the vendored mlx-lm fixes in patches/ (see patches/README.md).
# They live in .venv/, so `uv sync` reverts them; this is idempotent.
patch-venv:
	@./scripts/apply_patches.sh
.PHONY: patch-venv
