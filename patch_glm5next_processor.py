#!/usr/bin/env python3
"""Fail-closed build-time patch for the pinned vLLM GLM5 processor."""
from pathlib import Path


PROCESSOR = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/"
    "processors/glm5next.py"
)
OLD = 'with open(os.path.join(model_path, "processor_config.json")) as f:'
NEW = (
    'with open(__import__("transformers.utils.hub", '
    'fromlist=["cached_file"]).cached_file(model_path, '
    '"processor_config.json", **kwargs)) as f:'
)


def main():
    source = PROCESSOR.read_text()
    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(
            f"unexpected pinned GLM5 processor source: expected 1 match, got {count}"
        )
    PROCESSOR.write_text(source.replace(OLD, NEW))
    print("patched pinned GLM5 processor Hub lookup")


if __name__ == "__main__":
    main()
