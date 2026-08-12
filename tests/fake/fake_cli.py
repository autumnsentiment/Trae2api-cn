import json
import sys
import time


def main():
    args = sys.argv[1:]
    stdin_data = sys.stdin.read() if not sys.stdin.isatty() else ""
    if stdin_data.strip():
        prompt = stdin_data
    elif args:
        prompt = args[-1]
    else:
        prompt = ""

    text = "fake reply: " + (prompt[:40] if prompt else "no prompt")
    first = {
        "message": {"content": [{"type": "text", "text": text}]},
    }
    final = {
        "message": {"content": [{"type": "text", "text": text}]},
        "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
    }

    sys.stdout.write(json.dumps(first, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    time.sleep(0.02)
    sys.stdout.write(json.dumps(final, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
