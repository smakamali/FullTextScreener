# External_code (legacy, not maintained)

Earlier interactive chatbot versions of the pipeline (Gradio UI): `LiteratureChatbot.py` (app version 0.7.0) and `LiteratureChatbotLight.py` (0.8.0).

They are kept for reference only and **do not run with the current requirements**:
- **llama-index:** they use `ReActAgent.from_tools`, which llama-index 0.13 and later removed. Run them only in an environment with llama-index 0.12.
- **Extra packages:** they need `gradio`, `docker` and `langchain`, which are listed separately at the end of `Requirements/requirements_nv.txt`.

The maintained pipeline is `FullTextScreener.py` in the repository root.
