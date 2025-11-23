from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "./models/Qwen3-VL-4B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

print("Loading model (this may take some time on CPU)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="cpu",
    torch_dtype="auto",
)

print("Loaded successfully.")
