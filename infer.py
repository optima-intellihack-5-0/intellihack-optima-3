from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="Lakshith-403/optima_qwen-3b-instruct-ai-research-finetune",
	filename="q4_model_final.gguf",
)

response = llm(
	"Hello, how are you?",
	max_tokens=100,
    temperature=0.3,
)

print(response)