PY ?= python
LAKEHOUSE_DIR := .lakehouse/streaming-lakehouse
export LAKEHOUSE_DB ?= $(CURDIR)/$(LAKEHOUSE_DIR)/dbt/target/lakehouse.duckdb
PROVIDER ?= ollama
MODEL ?=

.PHONY: install demo-data test eval eval-selftest ask server

install:            ## Install dependencies
	$(PY) -m pip install -r requirements-dev.txt

demo-data:          ## Build the lakehouse this copilot queries (clones streaming-lakehouse, runs its local pipeline)
	@test -d $(LAKEHOUSE_DIR) || git clone --depth 1 https://github.com/charan32/streaming-lakehouse $(LAKEHOUSE_DIR)
	$(PY) -m pip install -q -r $(LAKEHOUSE_DIR)/requirements.txt
	$(MAKE) -C $(LAKEHOUSE_DIR) local PY=$(PY)

test:               ## Unit, integration and harness tests (no LLM needed)
	$(PY) -m pytest -q

eval:               ## Score a real model:  make eval PROVIDER=ollama MODEL=qwen2.5:7b
	$(PY) -m evals.run --provider $(PROVIDER) $(if $(MODEL),--model $(MODEL),)

eval-selftest:      ## Prove the harness works with a perfect and a useless agent
	$(PY) -m evals.run --provider oracle
	$(PY) -m evals.run --provider dont-know

ask:                ## make ask Q="Which category made the most revenue?"
	$(PY) -m copilot.ask --trace --provider $(PROVIDER) $(if $(MODEL),--model $(MODEL),) "$(Q)"

server:             ## Run the MCP server on stdio
	$(PY) -m copilot.server
