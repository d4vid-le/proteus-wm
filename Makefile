.PHONY: help test eval not-agi sandbox mcp clean

help:
	@echo "make test      run the core pytest suite"
	@echo "make eval      run the consolidated eval scoreboard (quick)"
	@echo "make not-agi   run the AGI falsification suite"
	@echo "make sandbox   run one autonomous pass over ./sandbox"
	@echo "make mcp       start the MCP server on stdio"
	@echo "make clean     remove caches"

test:
	python3 -m pytest tests/test_core.py -q

eval:
	python3 -m proteus.eval.run_evals --quick

not-agi:
	python3 -m proteus.eval.not_agi --quick

sandbox:
	python3 -m proteus.agent.autonomous --dir sandbox --once

mcp:
	python3 -m proteus.bridge.mcp_server

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find . -name ".DS_Store" -delete
