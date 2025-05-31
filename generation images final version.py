import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import uuid

# === Модель Stable Diffusion (CPU) ===
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe.to("cpu")

# === Папка для истории изображений ===
os.makedirs("history", exist_ok=True)

# === Функция генерации изображения ===
def generate(prompt, resolution, steps):
    if not prompt:
        return None, [], "Введите запрос..."

    try:
        width, height = map(int, resolution.split("x"))

        # Генерация с заданным числом шагов
        image = pipe(prompt, num_inference_steps=steps).images[0]
        image = image.resize((width, height))

        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join("history", filename)
        image.save(path)

        files = sorted(
            [os.path.join("history", f) for f in os.listdir("history")],
            key=os.path.getmtime,
            reverse=True
        )
        gallery = [Image.open(f) for f in files[:6]]
        return image, gallery, f"Готово! Кол-во шагов: {steps}"
    except Exception as e:
        return None, [], f"Ошибка: {str(e)}"

# === Список разрешений изображения (дополнен) ===
res_options = [
    "256x256", "384x384", "512x512", "640x360", "768x432",
    "800x800", "1024x576", "1280x720", "1600x900", "1920x1080"
]

# === Кастомный CSS (тёмная тема + стиль) ===
custom_css = """
body {
    background-color: #0d0d0d;
}
h1, label, .gr-textbox, .gr-dropdown, .gr-button, .gr-slider {
    color: white !important;
    font-family: 'Rubik', sans-serif;
}
.gr-button {
    background: linear-gradient(90deg, #007cf0, #00dfd8);
    color: white !important;
    font-size: 16px;
    padding: 12px 24px;
    border-radius: 8px;
}
.gr-image {
    border: 2px solid #00dfd8;
}
"""

# === Интерфейс Gradio ===
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Comadns #6</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Описание", placeholder="Пример: Лес в стиле неонового киберпанка")
            resolution = gr.Dropdown(res_options, value="512x512", label="Разрешение")
            steps = gr.Slider(5, 50, value=8, step=1, label="Кол-во шагов генерации")
            gr.Markdown(
                "**Что такое 'шаги генерации'?**<br>"
                "Больше шагов — выше качество, но дольше работает.<br>"
                "Рекомендуется: 8–25 на слабом ПК.",
                elem_id="steps-help"
            )
            generate_btn = gr.Button("Сгенерировать")
            status = gr.Textbox(label="", interactive=False)

        with gr.Column(scale=2):
            output = gr.Image(label="Результат", type="pil")

    gallery = gr.Gallery(label="История", columns=3, rows=2, height=200)

    generate_btn.click(
        fn=generate,
        inputs=[prompt, resolution, steps],
        outputs=[output, gallery, status]
    )

# === Запуск с публичной ссылкой ===
demo.launch(share=True)