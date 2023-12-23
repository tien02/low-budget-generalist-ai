CLIP_CKPT = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
CLIP_PORT = 8002
CLIP_API = f"http://0.0.0.0:{CLIP_PORT}/disease_information"

LLM_CKPT = "ckpt/llama-2-7b.Q4_K_M.gguf"
LLM_PORT = 8001
LLM_API = f"http://0.0.0.0:{LLM_PORT}/stream_tokens"

AGENT_PORT = 7860