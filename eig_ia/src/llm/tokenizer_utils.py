from typing import Any, Dict


def count_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(text.split()))


def merge_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "tokens_in": int(prompt_tokens),
        "tokens_out": int(completion_tokens),
        "tokens_total": int(prompt_tokens + completion_tokens),
    }
