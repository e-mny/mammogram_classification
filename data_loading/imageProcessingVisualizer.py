import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageEnhance, ImageTk

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)
        
        self.sharpness_scale = ttk.Scale(self.root, from_=0, to=2, orient="horizontal", length=200, label="Sharpness")
        self.sharpness_scale.set(1)  # Initial sharpness value
        self.sharpness_scale.pack(padx=10, pady=10)
        
        self.contrast_scale = ttk.Scale(self.root, from_=0, to=2, orient="horizontal", length=200, label="Contrast")
        self.contrast_scale.set(1)  # Initial contrast value
        self.contrast_scale.pack(padx=10, pady=10)
        
        self.load_button = ttk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(padx=10, pady=10)
        
        self.process_button = ttk.Button(self.root, text="Process Image", command=self.process_image)
        self.process_button.pack(padx=10, pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_image = Image.open(file_path)
            self.input_image_tk = ImageTk.PhotoImage(self.input_image)
            self.image_label.config(image=self.input_image_tk)
            
    def process_image(self):
        if hasattr(self, "input_image"):
            sharpness_factor = self.sharpness_scale.get()
            contrast_factor = self.contrast_scale.get()
            
            enhanced_image = self.input_image.copy()
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(sharpness_factor)
            
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(contrast_factor)
            
            self.output_image_tk = ImageTk.PhotoImage(enhanced_image)
            self.image_label.config(image=self.output_image_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
