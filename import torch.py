import torch
from diffusers import StableDiffusionPipeline

# Убедитесь, что мы используем CPU
device = "cpu"

# Загрузите модель
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to(device)

# Генерация изображения
prompt = "car"
image = pipeline(prompt).images[0]

# Сохранение изображения
image.save("generated_image.png")

print("Изображение сгенерировано и сохранено как generated_image.png")