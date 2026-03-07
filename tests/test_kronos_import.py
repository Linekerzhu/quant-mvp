from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("NeoQuasar/Kronos-small", trust_remote_code=True)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
