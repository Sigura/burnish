import json
import urllib.error
import urllib.request


class RefineError(RuntimeError):
    pass


def refine(server_url: str, text: str, mode: str = "all", model: str | None = None) -> dict:
    payload = {"text": text, "mode": mode}
    if model:
        payload["model"] = model
    request = urllib.request.Request(
        f"{server_url}/refine",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise RefineError(json.load(e).get("detail", str(e))) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RefineError(f"Backend not reachable at {server_url} ({e}). Try: systemctl --user status burnish") from e
