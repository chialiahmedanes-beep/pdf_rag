from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/llama-nemotron-rerank-vl-1b-v2",
    trust_remote_code=True,
    use_auth_token="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"
)

model = AutoModel.from_pretrained(
    "nvidia/llama-nemotron-rerank-vl-1b-v2",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
    use_auth_token="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"
)
