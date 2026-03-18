# OMEGA — Makefile for distributed experiment management (vast.ai / SSH)
#
# Usage:
#   make deploy         — deploy code + data to all servers
#   make run-ci         — run all pending experiments on servers (5 subjects)
#   make run            — run specific experiments: make run EXPS=exp_12,exp_13
#   make status         — check running experiments on servers
#   make logs S=name    — get logs from a server
#   make collect        — collect results from servers & update Qdrant (EXPS=exp_12,exp_15)
#   make summary        — print results summary
#   make test-local     — run exp_1 locally with CI subjects

SCRIPTS_DIR := scripts

.PHONY: deploy deploy-code deploy-data run run-ci run-all status logs kill kill-all collect summary test-local help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

deploy:  ## Deploy code + data to all remote servers
	bash $(SCRIPTS_DIR)/deploy_to_server.sh

deploy-code:  ## Deploy only code + install deps (no data)
	bash $(SCRIPTS_DIR)/deploy_to_server.sh --code-only

deploy-data:  ## Deploy only data to servers
	bash $(SCRIPTS_DIR)/deploy_to_server.sh --data-only

run:  ## Run experiments: make run EXPS=exp_12,exp_13 [S=server] [SUBJ=DB2_s1,...]
	python $(SCRIPTS_DIR)/run_distributed.py \
		--experiments $(EXPS) \
		$(if $(S),--server $(S)) \
		$(if $(SUBJ),--subjects $(SUBJ)) \
		$(if $(CI),--ci)

run-ci:  ## Run all pending experiments with CI test subjects (5 subj)
	python $(SCRIPTS_DIR)/run_distributed.py --all-pending --ci

run-all:  ## Run all pending experiments with full subject list
	python $(SCRIPTS_DIR)/run_distributed.py --all-pending

status:  ## Check running experiment status on servers
	python $(SCRIPTS_DIR)/run_distributed.py --status

logs:  ## Get logs from a server: make logs S=gpu-server-1 [EXP=exp_12]
	python $(SCRIPTS_DIR)/run_distributed.py --logs $(S) $(if $(EXP),--log-exp $(EXP))

kill:  ## Kill specific experiments: make kill EXPS=exp_7,exp_11 [S=server]
	python $(SCRIPTS_DIR)/run_distributed.py --kill $(EXPS) $(if $(S),--server $(S))

kill-all:  ## Kill ALL running experiments on all servers
	python $(SCRIPTS_DIR)/run_distributed.py --kill-all

collect:  ## Collect results from servers (EXPS=exp_12,exp_15 to filter) [S=server]
	python $(SCRIPTS_DIR)/collect_results.py $(if $(EXPS),--exp $(EXPS)) $(if $(S),--server $(S))

summary:  ## Print results summary
	python $(SCRIPTS_DIR)/collect_results.py --summary

test-local:  ## Run exp_1 locally with CI subjects (quick test)
	python experiments/exp_1_deep_raw_cnn_loso.py --ci
