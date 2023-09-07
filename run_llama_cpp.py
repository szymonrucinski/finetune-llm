from llama_cpp import Llama

llm = Llama(model_path="./models/gguf/krakowiak-7b.gguf.q4_k_m.bin")
output = llm(
    "### Użytkownik: Nazwij planety w naszym systemie slonecznym? ### Asystent: ",
    max_tokens=64,
    stop=["### Asystent: ", "\n"],
    echo=True,
)
print(output)
