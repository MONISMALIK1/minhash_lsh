.PHONY: help test install clean dedup

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

test:		## Run the offline test suite (no API key needed)
	cd .. && python -m unittest discover -s minhash_lsh/tests -t . -v

install:	## Editable install of the package
	pip install -e .

dedup:		## Find near-dupes: make dedup ARGS='corpus.jsonl --clusters'
	python -m minhash_lsh $(ARGS)

clean:		## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist .eggs
