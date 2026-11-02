"""Local development entry point for the shared Gemini + Ollama API."""

from cloud_api import app


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
