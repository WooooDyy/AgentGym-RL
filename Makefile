.PHONY: help test-docker docker-build docker-build-base docker-build-train docker-build-scripts docker-build-eval docker-train-shell docker-scripts-shell docker-env docker-eval docker-status docker-down docker-clean venv install setup-env start-env stop-env eval-env clean

ENV ?= searchqa
ENV_PORT ?= 36001
MODEL ?= gpt-4o-mini
MAX_ROUND ?= 10
LOCAL_DIR ?= saves/checkpoint
INFERENCE_FILE ?= $(ENV)_eval_sample.json
OUTPUT_DIR ?= ./$(ENV)/eval_results_$(ENV)
OPENAI_API_KEY ?= $(shell echo $$OPENAI_API_KEY)
OPENAI_BASE_URL ?= https://api.openai.com/v1
PROFILE ?= train

help:
	@echo "AgentGym-RL Makefile"
	@echo ""
	@echo "Docker: docker-build, docker-train-shell, docker-env, docker-eval, docker-status, docker-down, docker-clean, test-docker"
	@echo "Local:  venv, install, setup-env, start-env, stop-env, eval-env, clean"

test-docker:
	@docker --version && docker compose version
	@docker compose config --quiet && echo "docker-compose.yml: OK"
	@test -f docker/base.Dockerfile && echo "base.Dockerfile: OK"
	@test -f docker/train.Dockerfile && echo "train.Dockerfile: OK"
	@test -f docker/scripts.Dockerfile && echo "scripts.Dockerfile: OK"
	@test -f Dockerfile.eval && echo "Dockerfile.eval: OK"
	@test -f .dockerignore && echo ".dockerignore: OK"
	@test -f .env.example && echo ".env.example: OK"
	@echo "All tests PASSED"

docker-build: docker-build-base docker-build-train docker-build-scripts docker-build-eval

docker-build-base:
	docker build -f docker/base.Dockerfile -t agentgym-rl/base:latest .

docker-build-train: docker-build-base
	docker build -f docker/train.Dockerfile -t agentgym-rl/train:latest .

docker-build-scripts: docker-build-base
	docker build -f docker/scripts.Dockerfile -t agentgym-rl/scripts:latest .

docker-build-eval:
	docker build -f Dockerfile.eval -t agentgym/eval:latest .

docker-train-shell:
	@docker compose --profile train exec train /bin/bash 2>/dev/null || docker compose --profile train run --rm train /bin/bash

docker-scripts-shell:
	@docker compose --profile scripts exec scripts /bin/bash 2>/dev/null || docker compose --profile scripts run --rm scripts /bin/bash

docker-env:
	ENV=$(ENV) ENV_PORT=$(ENV_PORT) docker compose --profile env up -d env-server

docker-eval:
	ENV=$(ENV) docker compose --profile eval up --abort-on-container-exit eval-runner

docker-status:
	@docker ps --filter "name=agentgym" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || true
	@docker images | grep -E "^agentgym|REPOSITORY" || true

docker-down:
	docker compose down --remove-orphans

docker-clean:
	-docker rmi agentgym-rl/base:latest agentgym-rl/train:latest agentgym-rl/scripts:latest agentgym/eval:latest 2>/dev/null

venv:
	@test -d ".venv" && echo "venv exists" || uv venv

install: venv
	uv pip install -e AgentGym/agentenv

setup-env: install
	uv pip install -e AgentGym/agentenv-$(ENV)
	cd AgentGym/agentenv-$(ENV) && (test -f setup.sh && bash ./setup.sh || true)

start-env: venv
	uv run $(ENV) --host 0.0.0.0 --port $(ENV_PORT)

stop-env:
	@pkill -f "$(ENV) --host" || echo "No server running"

eval-env: venv
	@test -n "$(OPENAI_API_KEY)" || (echo "OPENAI_API_KEY required" && exit 1)
	cd $(ENV) && uv run python eval_$(ENV).py --inference_file $(INFERENCE_FILE) --output_dir $(OUTPUT_DIR) --model $(MODEL) --max_round $(MAX_ROUND) --api_key $(OPENAI_API_KEY) --base_url $(OPENAI_BASE_URL) --env_server_base http://localhost:$(ENV_PORT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
