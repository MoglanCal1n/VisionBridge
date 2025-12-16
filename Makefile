config ?= finetune_trafic.yaml

evaluate:
	uv run modal run src.street_object_detection.evaluate::main --config-file-name $(config)

fine-tune:
	uv run modal run src.street_object_detection.fine_tune::main --config-file-name $(config)

lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

install:
	uv sync