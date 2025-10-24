# AGENTS.md - Coding Standards & Commands

## Environment & Execution
- **Python**: Always use `uv run python <script>` (never bare `python` or `python3`)
- **Version**: Python 3.11+ required
- **Tests**: No test framework configured; test manually with `uv run python scripts/`
- **Linting/Formatting**: No formal lint/typecheck setup; follow code style conventions below

## Code Style
- **Imports**: Standard library → third-party → local (alphabetically within groups)
- **Docstrings**: Use Google-style docstrings for functions (see `scripts/predict.py` for examples)
- **Formatting**: Follow Python conventions (4-space indent, snake_case functions, PascalCase classes)
- **Error handling**: Use try/except with specific exception types; add informative error messages
- **Type hints**: Not required but encouraged for complex functions
- **Naming**: Descriptive names; avoid abbreviations except established conventions (cfg, np, cv2, etc.)

## Critical Project Rules
- **DO NOT READ `dataset/` FOLDERS** - Very large, will overflow context window
- **DO NOT "fix" COCO annotations without testing** - Category IDs starting from 0 are valid; Detectron2 handles mapping automatically
- **Always set `cfg.MODEL.DEVICE = "cpu"` for Mac** - No CUDA support, Detectron2 doesn't support MPS yet
- **Test inference with multiple confidence thresholds** (0.05, 0.1, 0.3, 0.5) - Undertrained models produce low scores
- **Training iterations**: 500 = demo only; 2000-3000 = production quality; check `fg_cls_accuracy` and `loss_cls` metrics