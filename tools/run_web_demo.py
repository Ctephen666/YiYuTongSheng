from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_web_config() -> dict:
    path = PROJECT_ROOT / "configs" / "web_demo.yaml"
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    import uvicorn

    config = _load_web_config()
    web = config.get("web", {}) if isinstance(config.get("web", {}), dict) else {}
    host = str(web.get("host", "127.0.0.1"))
    port = int(web.get("port", 7860))
    uvicorn.run("src.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
