import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

results = {}

# 1. Gemini
gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
if gemini_key and not gemini_key.startswith("your_"):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "VLM-DENTAL"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "").replace("models/", "") for m in data.get("models", [])]
            results["Gemini"] = {
                "status": "VALID",
                "total_models": len(models),
                "has_3_5_flash_lite": any("gemini-3.5-flash-lite" in m or "flash-lite" in m for m in models),
                "sample_models": [m for m in models if "flash" in m][:3],
            }
    except urllib.error.HTTPError as e:
        results["Gemini"] = {"status": f"HTTP_{e.code}", "error": e.reason}
    except Exception as e:
        results["Gemini"] = {"status": "ERROR", "error": str(e)}
else:
    results["Gemini"] = {"status": "NOT_CONFIGURED"}

# 2. NVIDIA NIM
nv_key = os.environ.get("NVIDIA_API_KEY", "").strip()
if nv_key and not nv_key.startswith("your_"):
    try:
        req = urllib.request.Request(
            "https://integrate.api.nvidia.com/v1/models",
            headers={"Authorization": f"Bearer {nv_key}", "User-Agent": "VLM-DENTAL"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id", "") for m in data.get("data", [])]
            results["NVIDIA_NIM"] = {
                "status": "VALID",
                "total_models": len(models),
                "has_muse_glimmer": "meta/muse-glimmer-30b" in models or any("glimmer" in m for m in models),
                "target_model_found": "meta/muse-glimmer-30b" in models,
            }
    except urllib.error.HTTPError as e:
        results["NVIDIA_NIM"] = {"status": f"HTTP_{e.code}", "error": e.reason}
    except Exception as e:
        results["NVIDIA_NIM"] = {"status": "ERROR", "error": str(e)}
else:
    results["NVIDIA_NIM"] = {"status": "NOT_CONFIGURED"}

# 3. Groq
groq_key = os.environ.get("GROQ_API_KEY", "").strip()
if groq_key and not groq_key.startswith("your_"):
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {groq_key}", "User-Agent": "VLM-DENTAL"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id", "") for m in data.get("data", [])]
            results["Groq"] = {
                "status": "VALID",
                "total_models": len(models),
                "has_qwen": any("qwen" in m.lower() for m in models),
                "qwen_models": [m for m in models if "qwen" in m.lower()][:3],
            }
    except urllib.error.HTTPError as e:
        results["Groq"] = {"status": f"HTTP_{e.code}", "error": e.reason}
    except Exception as e:
        results["Groq"] = {"status": "ERROR", "error": str(e)}
else:
    results["Groq"] = {"status": "NOT_CONFIGURED"}

# 4. OpenRouter
or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
if or_key and not or_key.startswith("your_"):
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {or_key}", "User-Agent": "VLM-DENTAL"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results["OpenRouter"] = {
                "status": "VALID",
                "key_info": data.get("data", {}),
            }
    except urllib.error.HTTPError as e:
        results["OpenRouter"] = {"status": f"HTTP_{e.code}", "error": e.reason}
    except Exception as e:
        results["OpenRouter"] = {"status": "ERROR", "error": str(e)}
else:
    results["OpenRouter"] = {"status": "NOT_CONFIGURED"}

print(json.dumps(results, indent=2))
