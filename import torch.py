import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

# Загрузка модели
device = "cpu"
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to(device)

def generate_image():
    prompt = prompt_entry.get()
    size = resolution_var.get()

    if not prompt:
        messagebox.showwarning("Пустой запрос", "Введите промпт для генерации.")
        return

    generate_button.config(state="disabled")
    save_button.config(state="disabled")
    progress_bar.start()
    result_label.config(text="Генерация изображения...")

    def worker():
        image = pipeline(prompt).images[0]
        width, height = map(int, size.split("x"))
        resized = image.resize((width, height))
        tk_image = ImageTk.PhotoImage(resized)

        def update_ui():
            image_label.config(image=tk_image)
            image_label.image = tk_image
            history_label.config(image=tk_image)
            history_label.image = tk_image
            result_label.config(text="Изображение сгенерировано.")
            save_button.config(state="normal")
            generate_button.config(state="normal")
            progress_bar.stop()

        image.save("last_generated.png")
        root.after(0, update_ui)

    threading.Thread(target=worker).start()

def save_image():
    filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png")])
    if filepath:
        image = Image.open("last_generated.png")
        image.save(filepath)
        messagebox.showinfo("Сохранено", f"Изображение сохранено как {filepath}")

# Интерфейс
root = tk.Tk()
root.title("Stable Diffusion Generator")
root.geometry("1920x1080")
root.configure(bg="#121212")

style = ttk.Style(root)
style.theme_use("default")
style.configure("TButton", font=("Arial", 16), padding=10)
style.configure("TLabel", background="#121212", foreground="white", font=("Arial", 14))
style.configure("TEntry", font=("Arial", 14))
style.configure("TProgressbar", thickness=20)

# Верхняя панель
top_frame = tk.Frame(root, bg="#121212")
top_frame.pack(anchor="w", padx=40, pady=20)

tk.Label(top_frame, text="Промпт:", bg="#121212", fg="white", font=("Arial", 18)).grid(row=0, column=0, sticky="w", pady=5)
prompt_entry = ttk.Entry(top_frame, width=60)
prompt_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(top_frame, text="Разрешение:", bg="#121212", fg="white", font=("Arial", 18)).grid(row=1, column=0, sticky="w", pady=5)

resolution_var = tk.StringVar(value="512x512")
res_options = [
    "256x256",
    "512x512",  # по умолчанию
    "768x768",
    "1024x1024",
    "1280x720",
    "1920x1080",
    "2048x2048",
    "3840x2160"
]
resolution_menu = ttk.OptionMenu(top_frame, resolution_var, res_options[1], *res_options)
resolution_menu.grid(row=1, column=1, sticky="w", pady=5)

generate_button = ttk.Button(top_frame, text="Сгенерировать", command=generate_image)
generate_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=15)

progress_bar = ttk.Progressbar(top_frame, mode="indeterminate", length=400)
progress_bar.grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

save_button = ttk.Button(top_frame, text="Сохранить изображение", command=save_image)
save_button.grid(row=4, column=0, columnspan=2, sticky="w", pady=10)
save_button.config(state="disabled")

result_label = ttk.Label(top_frame, text="")
result_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)

# Область изображения
display_frame = tk.Frame(root, bg="#1e1e1e", width=1024, height=1024, relief="solid", bd=2)
display_frame.pack(padx=40, pady=10, anchor="w")
image_label = tk.Label(display_frame, bg="#1e1e1e")
image_label.place(relx=0.5, rely=0.5, anchor="center")

# История последней генерации
tk.Label(root, text="Последнее изображение:", font=("Arial", 16), bg="#121212", fg="white").pack(anchor="w", padx=40)
history_label = tk.Label(root, bg="#121212")
history_label.pack(padx=40, anchor="w")

root.mainloop()