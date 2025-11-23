from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

model_path = "./models/Qwen3-VL-4B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="cpu",
    dtype="auto",
)

image = Image.open("images.jpeg").convert("RGB")
question = "Describe this image in one sentence."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=32)
output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("MODEL OUTPUT:")
print(output_text)
