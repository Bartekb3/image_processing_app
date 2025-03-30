import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------ PARAMETRY OKNA I WIELKOŚCI ------------------
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700

IMAGE_PREVIEW_SIZE = 350
IMAGE_PREVIEW_PADDING = 30  

OPTION_BUTTON_HEIGHT = 2
OPTION_BUTTON_WIDTH = 26

TOOLBAR_HEIGHT = 40
SIDEBAR_WIDTH = 200
SIDEBAR_INSIDE_BUTTON_WIDTH = 23
SIDEBAR_INSIDE_BUTTON_HEIGHT = 1

BOTTOM_BUTTON_PADDING = 15
BOTTOM_BUTTON_WIDTH = 15
BOTTOM_BUTTON_HEIGHT = 3

# ------------------ PARAMETRY OBSZARU WYKRESU ------------------
CHART_FRAME_WIDTH = 300   
CHART_FRAME_HEIGHT = 250  
CHART_FIG_WIDTH = 3       
CHART_FIG_HEIGHT = 2.5    

CONSTANT_SPACING_BETWEEN_IMAGES = 40

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(True, True)
        
        # ----- PRZECHOWYWANE DANE -----
        self.original_image = None  
        self.image_loaded_original = None
        self.current_processed_image = None  

        # ----- PARAMETRY / TOGGLES -----
        self.grayscale = False
        self.binarized = False
        self.negatived = False
        self.binarization_threshold = 128
        self.brightness = 0
        self.contrast = 1

        # Filtry
        self.apply_gauss_filter = False
        self.gauss_filter_sigma = 3
        
        self.apply_mean_filter = False
        self.mean_filter_sigma = 3
        
        self.apply_sharpen_filter = False
        self.sharpen_filter_sigma = 3
        
        # Filtr własny (custom)
        self.apply_custom_filter = False
        self.custom_filter_kernel = np.zeros((3,3), dtype=np.float32)
        
        # Wykrywanie krawędzi
        self.apply_edge_detection = False
        self.edge_detection_type = None  # "roberts" albo "sobel"

        # ----- TOOLBAR -----
        self.toolbar = tk.Frame(self.root, bg="#d9d9d9", height=TOOLBAR_HEIGHT)
        self.toolbar.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)
        
        self.load_button = tk.Button(
            self.toolbar, text="Wczytaj obraz", relief=tk.RAISED, padx=10, pady=5, command=self.load_image
        )
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_button = tk.Button(
            self.toolbar, text="Zapisz obraz", relief=tk.RAISED, padx=10, pady=5, command=self.save_image
        )
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.reset_button = tk.Button(
            self.toolbar, text="Ustawienia domyślne", relief=tk.RAISED, padx=10, pady=5, command=self.restore_default
        )
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.revert_button = tk.Button(
            self.toolbar, text="Cofnij wszystko", relief=tk.RAISED, padx=10, pady=5, command=self.revert_all
        )
        self.revert_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.apply_button = tk.Button(
            self.toolbar, text="Zastosuj", relief=tk.RAISED, padx=10, pady=5, command=self.apply_changes
        )
        self.apply_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.exit_button = tk.Button(
            self.toolbar, text="Wyjście", relief=tk.RAISED, padx=10, pady=5, command=self.root.quit
        )
        self.exit_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # ----- GŁÓWNA RAMKA -----
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # ----- SIDEBAR -----
        self.sidebar_frame = tk.Frame(self.main_frame, bg="#e0e0e0")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.sidebar_frame.configure(width=SIDEBAR_WIDTH)
        
        self.create_sidebar()

        # ----- OBRAZY + WYKRES -----
        self.image_display_frame = tk.Frame(self.main_frame, bg="#ffffff")
        self.image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 1) Kontener na obrazy
        self.images_container = tk.Frame(self.image_display_frame, bg="#ffffff")
        self.images_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 2) Ramka na wykres
        self.charts_frame = tk.Frame(
            self.image_display_frame, bg="#ddd",
            width=CHART_FRAME_WIDTH, height=CHART_FRAME_HEIGHT
        )
        self.charts_frame.pack(side=tk.TOP, pady=10)
        self.charts_frame.pack_propagate(False)
        self.chart_canvas = None

        # ----- Na starcie wyświetlamy placeholdery (3 kwadraty) -----
        self.display_placeholder_images()
        self.display_placeholder_chart()

    # =================== SIDEBAR I JEGO ZAWARTOŚĆ ===================
    def create_sidebar(self):
        self.sidebar_canvas = tk.Canvas(self.sidebar_frame, bg="#e0e0e0", width=SIDEBAR_WIDTH)
        self.scrollbar = tk.Scrollbar(self.sidebar_frame, orient=tk.VERTICAL, command=self.sidebar_canvas.yview)
        self.sidebar_content = tk.Frame(self.sidebar_canvas, bg="#e0e0e0", width=SIDEBAR_WIDTH)
        
        self.sidebar_content.bind(
            "<Configure>", lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        )
        
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw")
        self.sidebar_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Sekcje
        self.create_base_functions("Funkcje podstawowe")
        self.create_filters_functions("Filtry")
        self.create_custom_filter_section("Własny filtr")
        self.create_edge_detection_section("Wykrywanie krawędzi")
        self.create_plots_section("Wykresy")

    def toggle_section(self, frame):
        if frame.winfo_ismapped():
            frame.pack_forget()
        else:
            frame.pack(fill=tk.X)

    # ---------------------- PODSTAWOWE FUNKCJE ----------------------
    def create_base_functions(self, title):
        section_frame = tk.Frame(self.sidebar_content, bg="#cfcfcf", bd=2, relief=tk.GROOVE)
        section_frame.pack(fill=tk.X, pady=5)
        
        toggle_button = tk.Button(
            section_frame, text=title, height=OPTION_BUTTON_HEIGHT, width=OPTION_BUTTON_WIDTH,
            command=lambda: self.toggle_section(section_content)
        )
        toggle_button.pack(fill=tk.X)
        
        section_content = tk.Frame(section_frame, bg="#dcdcdc")
        section_content.pack(fill=tk.X)

        tk.Button(
            section_content, text="Skala szarości", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.change_grayscale
        ).pack(pady=5)

        tk.Button(
            section_content, text="Negatyw", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.change_negative
        ).pack(pady=5)

        tk.Button(
            section_content, text="Binaryzacja", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.change_binarization
        ).pack(pady=5)
        
        # Slider do binaryzacji
        self.binarize_threshold_slider = tk.Scale(section_content, from_=0, to=255, orient=tk.HORIZONTAL)
        self.binarize_threshold_slider.pack(fill=tk.X, padx=5, pady=5)
        self.binarize_threshold_slider.set(128)
        self.binarize_threshold_slider.bind("<ButtonRelease-1>", self.on_binarize_threshold_release)

        # Brightness
        tk.Label(section_content, text="Jasność").pack()
        self.brightness_slider = tk.Scale(section_content, from_=-255, to=255, resolution=1, orient=tk.HORIZONTAL)
        self.brightness_slider.pack(fill=tk.X, padx=5, pady=5)
        self.brightness_slider.set(0)
        self.brightness_slider.bind("<ButtonRelease-1>", self.on_brightness_release)

        # Contrast
        tk.Label(section_content, text="Kontrast").pack()
        self.contrast_slider = tk.Scale(section_content, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL)
        self.contrast_slider.pack(fill=tk.X, padx=5, pady=5)
        self.contrast_slider.set(1)
        self.contrast_slider.bind("<ButtonRelease-1>", self.on_contrast_release)

        section_content.pack_forget()

    # ---------------------- FILTRY (GAUSS, MEAN, SHARPEN) ----------------------
    def create_filters_functions(self, title):
        section_frame = tk.Frame(self.sidebar_content, bg="#cfcfcf", bd=2, relief=tk.GROOVE)
        section_frame.pack(fill=tk.X, pady=5)
        
        toggle_button = tk.Button(
            section_frame, text=title, height=OPTION_BUTTON_HEIGHT, width=OPTION_BUTTON_WIDTH,
            command=lambda: self.toggle_section(section_content)
        )
        toggle_button.pack(fill=tk.X)
        
        section_content = tk.Frame(section_frame, bg="#dcdcdc")
        section_content.pack(fill=tk.X)
        
        # Gauss
        tk.Button(
            section_content, text="Zastosuj filtr gaussa", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.add_gauss_filter
        ).pack(pady=5)

        tk.Label(section_content, text="Rozmiar filtru gaussa").pack()
        self.gauss_filter_sigma_slider = tk.Scale(section_content, from_=3, to=21, resolution=2, orient=tk.HORIZONTAL)
        self.gauss_filter_sigma_slider.pack(fill=tk.X, padx=5, pady=5)
        self.gauss_filter_sigma_slider.set(3)
        self.gauss_filter_sigma_slider.bind("<ButtonRelease-1>", self.on_gauss_sigma_release)

        # Mean
        tk.Button(
            section_content, text="Zastosuj filtr uśredniający", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.add_mean_filter
        ).pack(pady=5)

        tk.Label(section_content, text="Rozmiar filtru uśredniającego").pack()
        self.mean_filter_sigma_slider = tk.Scale(section_content, from_=3, to=21, resolution=2, orient=tk.HORIZONTAL)
        self.mean_filter_sigma_slider.pack(fill=tk.X, padx=5, pady=5)
        self.mean_filter_sigma_slider.set(3)
        self.mean_filter_sigma_slider.bind("<ButtonRelease-1>", self.on_mean_sigma_release)

        # Sharpen
        tk.Button(
            section_content, text="Zastosuj filtr wyostrzający", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.add_sharpen_filter
        ).pack(pady=5)

        tk.Label(section_content, text="Rozmiar filtru wyostrzającego").pack()
        self.sharpen_filter_sigma_slider = tk.Scale(section_content, from_=3, to=21, resolution=2, orient=tk.HORIZONTAL)
        self.sharpen_filter_sigma_slider.pack(fill=tk.X, padx=5, pady=5)
        self.sharpen_filter_sigma_slider.set(3)
        self.sharpen_filter_sigma_slider.bind("<ButtonRelease-1>", self.on_sharpen_sigma_release)

        section_content.pack_forget()

    # ---------------------- FILTR WŁASNY (CUSTOM) ----------------------
    def create_custom_filter_section(self, title):
        section_frame = tk.Frame(self.sidebar_content, bg="#cfcfcf", bd=2, relief=tk.GROOVE)
        section_frame.pack(fill=tk.X, pady=5)
        
        toggle_button = tk.Button(
            section_frame, text=title, height=OPTION_BUTTON_HEIGHT, width=OPTION_BUTTON_WIDTH,
            command=lambda: self.toggle_section(section_content)
        )
        toggle_button.pack(fill=tk.X)
        
        section_content = tk.Frame(section_frame, bg="#dcdcdc")
        section_content.pack(fill=tk.X)
        
        tk.Label(section_content, text="Wprowadź wartości kernela").pack(pady=5)
        self.custom_filter_entries = []
        for i in range(3):
            row_frame = tk.Frame(section_content, bg="#dcdcdc")
            row_frame.pack()
            row_entries = []
            for j in range(3):
                e = tk.Entry(row_frame, width=5)
                e.insert(0, "0")
                e.pack(side=tk.LEFT, padx=2, pady=2)
                row_entries.append(e)
            self.custom_filter_entries.append(row_entries)
        
        tk.Button(section_content, text="Ustaw filtr", command=self.set_custom_filter).pack(pady=5)
        tk.Button(section_content, text="Zastosuj filtr", command=self.toggle_custom_filter).pack(pady=5)
        
        section_content.pack_forget()

    # ---------------------- WYKRYWANIE KRAWĘDZI ----------------------
    def create_edge_detection_section(self, title):
        section_frame = tk.Frame(self.sidebar_content, bg="#cfcfcf", bd=2, relief=tk.GROOVE)
        section_frame.pack(fill=tk.X, pady=5)

        toggle_button = tk.Button(
            section_frame, text=title, height=OPTION_BUTTON_HEIGHT, width=OPTION_BUTTON_WIDTH,
            command=lambda: self.toggle_section(section_content)
        )
        toggle_button.pack(fill=tk.X)

        section_content = tk.Frame(section_frame, bg="#dcdcdc")
        section_content.pack(fill=tk.X)

        tk.Button(
            section_content, text="Krzyż Robertsa", width=SIDEBAR_INSIDE_BUTTON_WIDTH, height=SIDEBAR_INSIDE_BUTTON_HEIGHT,
            command=lambda: self.activate_edge_detection("roberts")
        ).pack(pady=5)

        tk.Button(
            section_content, text="Sobel", width=SIDEBAR_INSIDE_BUTTON_WIDTH, height=SIDEBAR_INSIDE_BUTTON_HEIGHT,
            command=lambda: self.activate_edge_detection("sobel")
        ).pack(pady=5)
        
        tk.Button(
            section_content, text="Wyłącz krawędzie", width=SIDEBAR_INSIDE_BUTTON_WIDTH, height=SIDEBAR_INSIDE_BUTTON_HEIGHT,
            command=lambda: self.activate_edge_detection(None)
        ).pack(pady=5)

        section_content.pack_forget()

    # ---------------------- SEKCJA: WYKRESY (HISTOGRAM, PROJEKCJE) ----------------------
    def create_plots_section(self, title):
        section_frame = tk.Frame(self.sidebar_content, bg="#cfcfcf", bd=2, relief=tk.GROOVE)
        section_frame.pack(fill=tk.X, pady=5)

        toggle_button = tk.Button(
            section_frame, text=title, height=OPTION_BUTTON_HEIGHT, width=OPTION_BUTTON_WIDTH,
            command=lambda: self.toggle_section(section_content)
        )
        toggle_button.pack(fill=tk.X)

        section_content = tk.Frame(section_frame, bg="#ddd")
        section_content.pack(fill=tk.X)

        tk.Button(
            section_content, text="Histogram", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.generate_histogram
        ).pack(pady=5)

        tk.Button(
            section_content, text="Proj. Pozioma", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.generate_horizontal_projection
        ).pack(pady=5)

        tk.Button(
            section_content, text="Proj. Pionowa", width=SIDEBAR_INSIDE_BUTTON_WIDTH,
            height=SIDEBAR_INSIDE_BUTTON_HEIGHT, command=self.generate_vertical_projection
        ).pack(pady=5)

        section_content.pack_forget()

    # =================== PLACEHOLDERY (3 KWADRATY) ===================
    def display_placeholder_images(self):
        """
        Creates two placeholder frames side-by-side, with a constant spacing in-between.
        They remain centered as a group in X-axis.
        """
        for widget in self.images_container.winfo_children():
            widget.destroy()

        two_image_frame = tk.Frame(self.images_container, bg="#ffffff")
        two_image_frame.pack(expand=True, anchor="center")  


        # LEFT PLACEHOLDER
        left_frame = tk.Frame(
            two_image_frame,
            width=IMAGE_PREVIEW_SIZE,
            height=int((2/3)*IMAGE_PREVIEW_SIZE),
            bg="#ddd",
            bd=0,
            relief=tk.FLAT
        )
        left_frame.pack(side=tk.LEFT, padx=(0, CONSTANT_SPACING_BETWEEN_IMAGES), pady=40)

        label_left = tk.Label(left_frame, text="Obraz oryginalny", bg="#ddd")
        label_left.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # RIGHT PLACEHOLDER
        right_frame = tk.Frame(
            two_image_frame,
            width=IMAGE_PREVIEW_SIZE,
            height=int((2/3)*IMAGE_PREVIEW_SIZE),
            bg="#ddd",
            bd=0,
            relief=tk.FLAT
        )
        right_frame.pack(side=tk.LEFT, padx=(0,0), pady=40)

        label_right = tk.Label(right_frame, text="Obraz przetworzony", bg="#ddd")
        label_right.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def display_placeholder_chart(self):
        """
        Tworzy placeholder (kwadrat) w self.charts_frame z napisem 'Wykres'.
        """
        self.clear_chart_if_exists()
        placeholder_label = tk.Label(self.charts_frame, text="Wykres", bg="#ddd")
        placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


    # =================== WYŚWIETLANIE OBRAZÓW PO WCZYTANIU ===================
    def display_images(self):
        """
        Displays the original & processed images side-by-side, with a constant spacing,
        and centers them as a group.
        """
        for widget in self.images_container.winfo_children():
            widget.destroy()

        if self.original_image is None:
            self.display_placeholder_images()
            return

        two_image_frame = tk.Frame(self.images_container, bg="#ffffff")
        two_image_frame.pack(expand=True, anchor="center")

        left_frame = tk.Frame(two_image_frame, width=IMAGE_PREVIEW_SIZE, height=IMAGE_PREVIEW_SIZE)
        left_frame.pack(side=tk.LEFT, padx=(0, CONSTANT_SPACING_BETWEEN_IMAGES), pady=20)

        right_frame = tk.Frame(two_image_frame, width=IMAGE_PREVIEW_SIZE, height=IMAGE_PREVIEW_SIZE)
        right_frame.pack(side=tk.LEFT, padx=(0,0), pady=20)

        processed_array = self.get_processed_image()
        self.current_processed_image = processed_array

        from PIL import Image
        image1 = Image.fromarray(self.original_image)
        image2 = Image.fromarray(processed_array)

        image1 = self.resize_preview(image1)
        image2 = self.resize_preview(image2)

        image1_tk = ImageTk.PhotoImage(image1)
        image2_tk = ImageTk.PhotoImage(image2)

        label1 = tk.Label(left_frame, image=image1_tk, bg="#ffffff")
        label1.image = image1_tk
        label1.pack()

        label2 = tk.Label(right_frame, image=image2_tk, bg="#ffffff")
        label2.image = image2_tk
        label2.pack()

    # =================== ŁADOWANIE / ZAPIS / RESET ===================
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if not file_path:
            return
        
        from PIL import Image
        image = Image.open(file_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        
        self.original_image = np.array(image, dtype=np.uint8)
        self.image_loaded_original = self.original_image.copy()
        self.display_images()

    def save_image(self):
        if self.original_image is None:
            return
        processed = self.current_processed_image
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp")]
        )
        if file_path:
            from PIL import Image
            out_img = Image.fromarray(processed)
            out_img.save(file_path)

    def apply_changes(self):
        image = self.get_processed_image()
        self.original_image = image
        self.restore_default()
        self.display_images()

    def revert_all(self):
        if self.image_loaded_original is not None:
            self.original_image = self.image_loaded_original.copy()
            self.restore_default()
            self.display_images()

    def restore_default(self):
        self.grayscale = False
        self.binarized = False
        self.negatived = False
        self.binarization_threshold = 128
        self.brightness = 0
        self.contrast = 1
        self.apply_gauss_filter = False
        self.gauss_filter_sigma = 3
        self.apply_mean_filter = False
        self.mean_filter_sigma = 3
        self.apply_sharpen_filter = False
        self.sharpen_filter_sigma = 3
        self.apply_custom_filter = False
        self.custom_filter_kernel = np.zeros((3,3), dtype=np.float32)
        self.apply_edge_detection = False
        self.edge_detection_type = None

        self.binarize_threshold_slider.set(128)
        self.brightness_slider.set(0)
        self.contrast_slider.set(1)
        self.gauss_filter_sigma_slider.set(3)
        self.mean_filter_sigma_slider.set(3)
        self.sharpen_filter_sigma_slider.set(3)

        self.display_images()

    # =================== OBLICZANIE OBRAZU PO PRZETWORZENIU ===================
    def get_processed_image(self):
        image = self.original_image.copy().astype(np.float32)

        # 1. Grayscale
        if self.grayscale:
            image = self.convert_to_grayscale(image)

        # 2. Binarize
        if self.binarized:
            if len(image.shape) == 3:
                image = self.convert_to_grayscale(image)
            image = self.binarize(image, self.binarization_threshold)

        # 3. Negative
        if self.negatived:
            image = self.negative(image)

        # 4. Brightness + Contrast
        image = self.apply_brightness(image, self.brightness)
        image = self.adjust_contrast(image, self.contrast)

        # 5. Filters
        if self.apply_gauss_filter:
            image = self.apply_convolution(image, self.gauss_filter(), self.gauss_filter_sigma)
        if self.apply_mean_filter:
            image = self.apply_convolution(image, self.mean_filter(), self.mean_filter_sigma)
        if self.apply_sharpen_filter:
            image = self.apply_convolution(image, self.sharpen_filter(), self.sharpen_filter_sigma)
        if self.apply_custom_filter and self.custom_filter_kernel is not None:
            image = self.apply_convolution(image, self.custom_filter_kernel, 3)

        # 6. Edge detection
        if self.apply_edge_detection and self.edge_detection_type is not None:
            if self.edge_detection_type == "roberts":
                image = self.apply_roberts_cross(image)
            elif self.edge_detection_type == "sobel":
                image = self.apply_sobel(image)

        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    # ---------------------- PRZETWARZANIE PODSTAWOWE ----------------------
    def convert_to_grayscale(self, img_matrix):
        if len(img_matrix.shape) == 3:
            gray = np.dot(img_matrix[:, :, :3], [0.299, 0.587, 0.114])
            return gray
        else:
            return img_matrix

    def binarize(self, img_matrix, threshold):
        binarized_matrix = np.where(img_matrix > threshold, 255, 0).astype(np.float32)
        return binarized_matrix

    def negative(self, img_matrix):
        return 255 - img_matrix

    def apply_brightness(self, img_matrix, brightness):
        return img_matrix + brightness

    def adjust_contrast(self, image, factor=1):
        mean = np.mean(image)
        adjusted = mean + (image - mean) * factor
        return adjusted

    # ---------------------- KONWOLUCJA ----------------------
    def apply_convolution(self, image, kernel, _size=3):
        img_in = image.astype(np.float32)

        if len(img_in.shape) == 3:  # RGB
            R = img_in[:, :, 0]
            G = img_in[:, :, 1]
            B = img_in[:, :, 2]

            Rf = self.convolve2d(R, kernel)
            Gf = self.convolve2d(G, kernel)
            Bf = self.convolve2d(B, kernel)

            out = np.stack((Rf, Gf, Bf), axis=-1)
        else:
            out = self.convolve2d(img_in, kernel)

        return out

    def convolve2d(self, channel, kernel):
        h, w = channel.shape
        kh, kw = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2

        s = np.sum(kernel)
        if s != 0:
            kernel = kernel / s

        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(channel, dtype=np.float32)

        for y in range(h):
            for x in range(w):
                region = padded[y:y+kh, x:x+kw]
                val = np.sum(region * kernel)
                result[y, x] = val

        return result

    # ---------------------- FILTRY GOTOWE (MEAN, GAUSS, SHARPEN) ----------------------
    def mean_filter(self):
        size = self.mean_filter_sigma
        if size < 3:
            size = 3
        kernel = np.ones((size, size), dtype=np.float32)
        return kernel

    def gauss_filter(self):
        size = self.gauss_filter_sigma
        if size < 3:
            size = 3

        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        g_sigma = size / 3.0
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2*g_sigma*g_sigma))
        return kernel

    def sharpen_filter(self):
        size = self.sharpen_filter_sigma
        if size < 3:
            size = 3

        if size == 3:
            kernel = np.array([
                [ 0, -1,  0],
                [-1,  5, -1],
                [ 0, -1,  0]
            ], dtype=np.float32)
        else:
            kernel = np.full((size, size), -1, dtype=np.float32)
            kernel[size//2, size//2] = size * size
        return kernel

    # ---------------------- FILTR WŁASNY ----------------------
    def set_custom_filter(self):
        kernel = []
        for row_entries in self.custom_filter_entries:
            row_vals = []
            for entry in row_entries:
                try:
                    val = float(entry.get())
                except ValueError:
                    val = 0.0
                row_vals.append(val)
            kernel.append(row_vals)
        kernel = np.array(kernel, dtype=np.float32)
        self.custom_filter_kernel = kernel
        
        if self.apply_custom_filter:
            self.display_images()

    def toggle_custom_filter(self):
        self.apply_custom_filter = not self.apply_custom_filter
        self.display_images()

    # ---------------------- WYKRYWANIE KRAWĘDZI (ROBERTS, SOBEL) ----------------------
    def activate_edge_detection(self, method):
        if method is None:
            self.apply_edge_detection = False
            self.edge_detection_type = None
        else:
            self.apply_edge_detection = True
            self.edge_detection_type = method
        self.display_images()

    def apply_roberts_cross(self, image):
        if len(image.shape) == 3:
            image = self.convert_to_grayscale(image)
        gx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        gy = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        ix = self.convolve2d(image, gx)
        iy = self.convolve2d(image, gy)
        grad = np.sqrt(ix**2 + iy**2)
        return grad

    def apply_sobel(self, image):
        if len(image.shape) == 3:
            image = self.convert_to_grayscale(image)

        gx = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ], dtype=np.float32)
        gy = np.array([
            [ 1,2,1],
            [ 0,0,0],
            [-1,-2,-1]
        ], dtype=np.float32)

        ix = self.convolve2d(image, gx)
        iy = self.convolve2d(image, gy)
        grad = np.sqrt(ix**2 + iy**2)
        return grad

    # ---------------------- OBSŁUGA SLIDERÓW ----------------------
    def on_binarize_threshold_release(self, event):
        self.binarization_threshold = int(self.binarize_threshold_slider.get())
        if self.binarized:
            self.display_images()

    def on_brightness_release(self, event):
        self.brightness = int(self.brightness_slider.get())
        self.display_images()

    def on_contrast_release(self, event):
        self.contrast = float(self.contrast_slider.get())
        self.display_images()

    def on_gauss_sigma_release(self, event):
        self.gauss_filter_sigma = int(self.gauss_filter_sigma_slider.get())
        if self.apply_gauss_filter:
            self.display_images()

    def on_mean_sigma_release(self, event):
        self.mean_filter_sigma = int(self.mean_filter_sigma_slider.get())
        if self.apply_mean_filter:
            self.display_images()

    def on_sharpen_sigma_release(self, event):
        self.sharpen_filter_sigma = int(self.sharpen_filter_sigma_slider.get())
        if self.apply_sharpen_filter:
            self.display_images()

    # ---------------------- TOGGLES PRZYCISKÓW ----------------------
    def change_grayscale(self):
        self.grayscale = not self.grayscale
        self.display_images()

    def change_negative(self):
        self.negatived = not self.negatived
        self.display_images()

    def change_binarization(self):
        self.binarized = not self.binarized
        self.display_images()

    def add_gauss_filter(self):
        self.apply_gauss_filter = not self.apply_gauss_filter
        self.display_images()

    def add_mean_filter(self):
        self.apply_mean_filter = not self.apply_mean_filter
        self.display_images()

    def add_sharpen_filter(self):
        self.apply_sharpen_filter = not self.apply_sharpen_filter
        self.display_images()

    # ---------------------- WYKRESY (HISTOGRAM, PROJEKCJE) ----------------------
    def clear_chart_if_exists(self):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
            self.chart_canvas = None

    def generate_histogram(self):
        self.clear_chart_if_exists()
        processed = self.get_processed_image()

        fig = plt.Figure(figsize=(CHART_FIG_WIDTH, CHART_FIG_HEIGHT))
        ax = fig.add_subplot(111)

        if len(processed.shape) == 2:
            # Grayscale
            hist_vals = np.zeros(256, dtype=np.int32)
            flat = processed.flatten()
            for v in flat:
                hist_vals[v] += 1
            hist_vals = hist_vals / np.sum(hist_vals)  
            ax.bar(range(256), hist_vals)
            ax.set_title("Histogram (Grayscale)")
        else:
            # Color channels
            for c, color_name in zip(range(3), ["R", "G", "B"]):
                hist_vals = np.zeros(256, dtype=np.int32)
                flat = processed[:, :, c].flatten()
                for v in flat:
                    hist_vals[v] += 1
                hist_vals = hist_vals / np.sum(hist_vals)
                ax.plot(range(256), hist_vals, label=color_name)
            ax.legend()
            ax.set_title("Histogram (RGB)")

        ax.set_xlim([0, 255])
        fig.tight_layout()

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(expand=True, fill="both")

    def generate_horizontal_projection(self):
        self.clear_chart_if_exists()
        processed = self.get_processed_image()
        
        if len(processed.shape) == 3:
            processed = self.convert_to_grayscale(processed)

        horizontal_proj = np.sum(processed, axis=1)
        horizontal_proj = horizontal_proj / np.max(horizontal_proj)

        fig = plt.Figure(figsize=(CHART_FIG_WIDTH, CHART_FIG_HEIGHT))
        ax = fig.add_subplot(111)
        ax.plot(range(len(horizontal_proj)), horizontal_proj)
        ax.set_title("Horizontal Projection")

        fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(expand=True, fill="both")

    def generate_vertical_projection(self):
        self.clear_chart_if_exists()
        processed = self.get_processed_image()
        
        if len(processed.shape) == 3:
            processed = self.convert_to_grayscale(processed)

        vertical_proj = np.sum(processed, axis=0)
        vertical_proj = vertical_proj / np.max(vertical_proj)

        fig = plt.Figure(figsize=(CHART_FIG_WIDTH, CHART_FIG_HEIGHT))
        ax = fig.add_subplot(111)
        ax.plot(range(len(vertical_proj)), vertical_proj)
        ax.set_title("Vertical Projection")

        fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(expand=True, fill="both")

    # ---------------------- ROZMIAR PODGLĄDU ----------------------
    def resize_preview(self, pil_image):
        width, height = pil_image.size
        ratio = min(IMAGE_PREVIEW_SIZE / width, IMAGE_PREVIEW_SIZE / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return pil_image.resize((new_width, new_height), Image.LANCZOS)

# ======================= MAIN =======================
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
