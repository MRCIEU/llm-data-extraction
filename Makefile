.PHONY: clean data lint docs

archive_host:=epi-franklin2
archive_path:=/projects/MRC-IEU/research/projects/ieu3/p3/015/working/data/llm-data-extraction/
data_archive:=$(archive_host):$(archive_path)

#################################################################################
# Rules
#################################################################################

## ==== sanity-check ====
check-health:
	echo "Data archive data_archive: " $(data_archive)
	pip list | grep local_funcs
	pip list | grep yiutils

## ==== data ====

## data sync (dry run)
data-sync-dry:
	rsync -aLvzP -n ./data $(data_archive)

## data sync
data-sync:
	rsync -aLvzP ./data $(data_archive)

## ==== docs ====

## docs-all: all docs
docs-all: docs-data-archive docs-filetree

## docs-filetree: show filetree
docs-filetree:
	OUTFILE=./docs/filetree.txt; \
	eza -T --git-ignore ./ > $${OUTFILE}

## docs-data-archive: show details on data archive; need vpn
docs-data-archive:
	OUTFILE=./docs/data_archive.txt; \
	ssh $(archive_host) "eza -T -L 3 $(archive_path)" > $${OUTFILE}

## ==== codebase ====

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}'
