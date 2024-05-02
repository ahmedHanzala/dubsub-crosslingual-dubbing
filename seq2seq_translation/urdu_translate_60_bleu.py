from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./models/ur-en-tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/ur-en-model")

timestamp_special_tokens = str(["[S]", "[E]"])
def translate(text,timestamps):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    if timestamps:
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=timestamp_special_tokens)
    else:
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text , timestamps
