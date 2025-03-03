import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk  # Tylko do wczytania/zapisu i wyświetlania

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prosta aplikacja do przetwarzania obrazów (na 4)")

        # Obraz oryginalny i aktualny (jako numpy array)
        self.original_image_array = None
        self.current_image_array = None

        # Panel na przyciski i suwaki
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Panel do wyświetlania obrazu
        self.image_label = tk.Label(root)
        self.image_label.pack(side=tk.RIGHT, padx=5, pady=5)

        # --- PRZYCISKI PLIK ---
        btn_load = tk.Button(control_frame, text="Wczytaj obraz", command=self.load_image)
        btn_load.pack(pady=2, fill=tk.X)

        btn_save = tk.Button(control_frame, text="Zapisz obraz", command=self.save_image)
        btn_save.pack(pady=2, fill=tk.X)

        btn_reset = tk.Button(control_frame, text="Reset do oryginału", command=self.reset_image)
        btn_reset.pack(pady=2, fill=tk.X)

        # --- SUWAK JASNOŚĆ ---
        self.brightness_scale = tk.Scale(control_frame, from_=-128, to=128, 
                                         orient=tk.HORIZONTAL, label="Jasność",
                                         command=self.change_brightness)
        self.brightness_scale.pack(pady=2, fill=tk.X)

        # --- SUWAK KONTRAST ---
        self.contrast_scale = tk.Scale(control_frame, from_=-128, to=128,
                                       orient=tk.HORIZONTAL, label="Kontrast",
                                       command=self.change_contrast)
        self.contrast_scale.pack(pady=2, fill=tk.X)

        # --- SUWAK PROG BINARYZACJI ---
        self.threshold_scale = tk.Scale(control_frame, from_=0, to=255,
                                        orient=tk.HORIZONTAL, label="Próg binar.",
                                        command=self.binarize)
        self.threshold_scale.pack(pady=2, fill=tk.X)

        # --- PRZYCISKI OPERACJI ---
        btn_gray = tk.Button(control_frame, text="Konwersja do szarości", command=self.convert_to_gray)
        btn_gray.pack(pady=2, fill=tk.X)

        btn_negative = tk.Button(control_frame, text="Negatyw", command=self.negative)
        btn_negative.pack(pady=2, fill=tk.X)

        # --- FILTRY ---
        lbl_filt = tk.Label(control_frame, text="Filtry (3x3 przykłady):")
        lbl_filt.pack(pady=2)

        btn_mean = tk.Button(control_frame, text="Uśredniający", command=self.mean_filter)
        btn_mean.pack(pady=2, fill=tk.X)

        btn_gauss = tk.Button(control_frame, text="Gaussa (3x3)", command=self.gauss_filter)
        btn_gauss.pack(pady=2, fill=tk.X)

        btn_sharpen = tk.Button(control_frame, text="Wyostrzający", command=self.sharpen_filter)
        btn_sharpen.pack(pady=2, fill=tk.X)

        btn_custom = tk.Button(control_frame, text="Własny filtr", command=self.custom_filter_dialog)
        btn_custom.pack(pady=2, fill=tk.X)

        # --- HISTOGRAM, PROJEKCJE, KRAWĘDZIE ---
        btn_hist = tk.Button(control_frame, text="Histogram", command=self.show_histogram)
        btn_hist.pack(pady=2, fill=tk.X)

        btn_proj_h = tk.Button(control_frame, text="Projekcja pozioma", command=self.show_horizontal_projection)
        btn_proj_h.pack(pady=2, fill=tk.X)

        btn_proj_v = tk.Button(control_frame, text="Projekcja pionowa", command=self.show_vertical_projection)
        btn_proj_v.pack(pady=2, fill=tk.X)

        btn_edges = tk.Button(control_frame, text="Krawędzie", command=self.show_edge_dialog)
        btn_edges.pack(pady=2, fill=tk.X)

    # ----------------------
    # Pomocnicze funkcje I/O
    # ----------------------
    def load_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not filename:
            return
        pil_img = Image.open(filename).convert("RGB")
        self.original_image_array = np.array(pil_img, dtype=np.uint8)
        self.current_image_array = self.original_image_array.copy()
        self.update_display()

    def save_image(self):
        if self.current_image_array is None:
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if not filename:
            return
        # Konwertujemy z numpy do PIL i zapisujemy
        pil_img = Image.fromarray(self.current_image_array)
        pil_img.save(filename)
        messagebox.showinfo("Zapisano", f"Zapisano do pliku: {filename}")

    def reset_image(self):
        if self.original_image_array is None:
            return
        self.current_image_array = self.original_image_array.copy()
        # resetujemy suwaki
        self.brightness_scale.set(0)
        self.contrast_scale.set(0)
        self.threshold_scale.set(127)
        self.update_display()

    def update_display(self):
        """Konwertuje current_image_array do ImageTk i wyświetla."""
        if self.current_image_array is None:
            return
        disp_img = Image.fromarray(self.current_image_array)
        disp_imgtk = ImageTk.PhotoImage(disp_img)
        self.image_label.config(image=disp_imgtk)
        self.image_label.image = disp_imgtk  # zachowanie referencji

    # ----------------------
    # Podstawowe operacje
    # ----------------------
    def convert_to_gray(self):
        if self.current_image_array is None:
            return
        # Stosujemy formułę 0.299*R + 0.587*G + 0.114*B
        if len(self.current_image_array.shape) == 3:  # RGB
            R = self.current_image_array[:, :, 0]
            G = self.current_image_array[:, :, 1]
            B = self.current_image_array[:, :, 2]
            gray = 0.299*R + 0.587*G + 0.114*B
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            # Zastępujemy obraz 2D zamiast 3D, ale by wyświetlać w RGB, tworzymy 3 kanały takie same
            self.current_image_array = np.stack((gray, gray, gray), axis=-1)
            self.update_display()

    def change_brightness(self, val):
        if self.current_image_array is None:
            return
        # najpierw reset na oryginał, potem nakładamy jasność i kontrast sekwencyjnie
        self.current_image_array = self.original_image_array.copy()

        # pobieramy wartości suwaków
        b = self.brightness_scale.get()
        # c = self.contrast_scale.get()

        # nakładamy jasność
        self.current_image_array = self.current_image_array.astype(np.int16)
        self.current_image_array += b
        self.current_image_array = np.clip(self.current_image_array, 0, 255)

        # nakładamy kontrast (w identyczny sposób, co w change_contrast)
        # self.apply_contrast_int16(c)

        # jeśli suwak binaryzacji ustawiony, też aktualizujemy
        # t = self.threshold_scale.get()
        # self.apply_threshold(t)

        self.current_image_array = self.current_image_array.astype(np.uint8)
        self.update_display()

    def change_contrast(self, val):
        if self.current_image_array is None:
            return
        # podobnie jak powyżej – zawsze bazujemy na oryginale + jasność + kontrast + progowanie
        self.current_image_array = self.original_image_array.copy()

        b = self.brightness_scale.get()
        c = self.contrast_scale.get()
        t = self.threshold_scale.get()

        self.current_image_array = self.current_image_array.astype(np.int16)
        self.current_image_array += b
        self.current_image_array = np.clip(self.current_image_array, 0, 255)

        self.apply_contrast_int16(c)

        self.apply_threshold(t)

        self.current_image_array = self.current_image_array.astype(np.uint8)
        self.update_display()

    def apply_contrast_int16(self, c):
        # c – wartość z suwaka w zakresie -128..128
        # Użyjemy formuły: newVal = (oldVal - 128) * s + 128
        # a s to (1 + c/128) (np. c=128 -> s=2.0)
        s = 1 + (c / 128.0)
        self.current_image_array -= 128
        self.current_image_array = self.current_image_array * s
        self.current_image_array += 128
        self.current_image_array = np.clip(self.current_image_array, 0, 255)

    def negative(self):
        if self.current_image_array is None:
            return
        # negatyw: 255 - pixel
        arr = self.current_image_array.astype(np.int16)
        arr = 255 - arr
        arr = np.clip(arr, 0, 255)
        self.current_image_array = arr.astype(np.uint8)
        self.update_display()

    def binarize(self, val):
        if self.current_image_array is None:
            return
        # podobnie – bazujemy na oryginale + jasność + kontrast, a potem prog
        self.current_image_array = self.original_image_array.copy()

        b = self.brightness_scale.get()
        c = self.contrast_scale.get()
        t = self.threshold_scale.get()

        self.current_image_array = self.current_image_array.astype(np.int16)
        self.current_image_array += b
        self.current_image_array = np.clip(self.current_image_array, 0, 255)
        self.apply_contrast_int16(c)
        self.apply_threshold(t)
        self.current_image_array = self.current_image_array.astype(np.uint8)
        self.update_display()

    def apply_threshold(self, t):
        # jeżeli aktualny obraz jest RGB, stosujemy prog na każdym kanale lub
        # typowo najpierw warto przejść do szarości, ale tu zrobimy prosto (per kanał).
        self.current_image_array = np.where(self.current_image_array < t, 0, 255)

    # ----------------------
    # Filtry splotowe
    # ----------------------
    def mean_filter(self):
        # 3x3 filtr uśredniający
        kernel = np.ones((3,3), dtype=np.float32)/9.0
        self.apply_convolution(kernel)

    def gauss_filter(self):
        # 3x3 Gauss
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32)
        kernel /= np.sum(kernel)
        self.apply_convolution(kernel)

    def sharpen_filter(self):
        # 3x3 wyostrzanie
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=np.float32)
        self.apply_convolution(kernel)

    def custom_filter_dialog(self):
        if self.current_image_array is None:
            return
        # Okno, w którym użytkownik wprowadzi 9 wag do 3x3
        top = tk.Toplevel(self.root)
        top.title("Własny filtr 3x3")

        entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                e = tk.Entry(top, width=5)
                e.grid(row=i, column=j, padx=2, pady=2)
                e.insert(0, "0")
                row_entries.append(e)
            entries.append(row_entries)

        def apply_custom():
            k = np.zeros((3,3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    try:
                        val = float(entries[i][j].get())
                    except ValueError:
                        val = 0.0
                    k[i,j] = val
            self.apply_convolution(k)
            top.destroy()

        btn_ok = tk.Button(top, text="Zastosuj", command=apply_custom)
        btn_ok.grid(row=3, column=1, pady=5)

    def apply_convolution(self, kernel):
        if self.current_image_array is None:
            return
        # zabezpieczamy się, żeby nie przekraczać typów
        img_in = self.current_image_array.astype(np.float32)

        # Jeśli jest RGB, konwolujemy każdy kanał osobno
        if len(img_in.shape) == 3:
            # Rozbijamy na kanały
            R = img_in[:, :, 0]
            G = img_in[:, :, 1]
            B = img_in[:, :, 2]

            Rf = self.convolve2d(R, kernel)
            Gf = self.convolve2d(G, kernel)
            Bf = self.convolve2d(B, kernel)

            out = np.stack((Rf, Gf, Bf), axis=-1)
        else:
            # Skala szarości
            out = self.convolve2d(img_in, kernel)

        out = np.clip(out, 0, 255).astype(np.uint8)
        self.current_image_array = out
        self.update_display()

    def convolve2d(self, channel, kernel):
        """Konwolucja 2D dla jednego kanału."""
        h, w = channel.shape
        kh, kw = kernel.shape
        pad_h = kh//2
        pad_w = kw//2

        # Tworzymy wynik i bufor z paddingiem
        result = np.zeros_like(channel, dtype=np.float32)
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        for y in range(h):
            for x in range(w):
                region = padded[y:y+kh, x:x+kw]
                val = np.sum(region * kernel)
                result[y,x] = val

        return result

    # ----------------------
    # Histogram
    # ----------------------
    def show_histogram(self):
        if self.current_image_array is None:
            return
        # Tworzymy nowe okno z Canvas
        hist_win = tk.Toplevel(self.root)
        hist_win.title("Histogram")
        canvas = tk.Canvas(hist_win, width=260, height=200, bg="white")
        canvas.pack()

        arr = self.current_image_array
        # Jeżeli RGB, rysujemy 3 histogramy w kanałach obok siebie lub kolejno
        # (tutaj dla uproszczenia zrobimy osobno w pionie).
        # Albo zrobimy jeden wspólny? Lepiej rozdzielić – ale to wymaga 3x tyle miejsca itd.
        # Poniżej przykład: jeśli to RGB, bierzemy składową szarości "mean" do narysowania pojedynczego histogramu
        # Dla pełnej dokładności można narysować 3 osobne. Tu dla prostoty: histogram ze średniej.

        if len(arr.shape) == 3:
            R = arr[:,:,0]
            G = arr[:,:,1]
            B = arr[:,:,2]
            gray = (R.astype(np.int32) + G.astype(np.int32) + B.astype(np.int32)) // 3
        else:
            gray = arr

        hist_vals, _ = np.histogram(gray, bins=256, range=(0,256))
        max_val = np.max(hist_vals)
        # Rysujemy słupki
        for i in range(256):
            v = hist_vals[i]
            # skalujemy do 200 px
            scaled = int((v / max_val) * 200)
            # rysujemy linię
            canvas.create_line(i+2, 200, i+2, 200 - scaled, fill="black")

    # ----------------------
    # Projekcje
    # ----------------------
    def show_horizontal_projection(self):
        if self.current_image_array is None:
            return
        # Obliczamy sumy w każdym wierszu (dla obrazu grayscale lub RGB – wtedy weźmy sumę kanałów)
        arr = self.to_grayscale(self.current_image_array)
        proj = np.sum(arr, axis=1)  # sumujemy w poziomie, czyli sumy w wierszach
        self.draw_projection(proj, "Projekcja pozioma")

    def show_vertical_projection(self):
        if self.current_image_array is None:
            return
        arr = self.to_grayscale(self.current_image_array)
        proj = np.sum(arr, axis=0)  # sumujemy w pionie, czyli sumy w kolumnach
        self.draw_projection(proj, "Projekcja pionowa")

    def to_grayscale(self, arr):
        if len(arr.shape) == 3:
            R = arr[:,:,0].astype(np.int32)
            G = arr[:,:,1].astype(np.int32)
            B = arr[:,:,2].astype(np.int32)
            gray = (0.299*R + 0.587*G + 0.114*B)
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            return gray
        else:
            return arr

    def draw_projection(self, proj, title):
        # Stworzymy okno i narysujemy wykres linii
        proj_win = tk.Toplevel(self.root)
        proj_win.title(title)
        canvas = tk.Canvas(proj_win, width=500, height=200, bg="white")
        canvas.pack()

        # Normalizujemy do height=200
        H = 200
        max_val = np.max(proj)
        if max_val == 0:
            max_val = 1
        scale = H / max_val

        # Rysujemy
        w = len(proj)
        last_x = 0
        last_y = H - int(proj[0]*scale)
        for i in range(1, w):
            x = i
            y = H - int(proj[i]*scale)
            canvas.create_line(last_x, last_y, x, y, fill="black")
            last_x, last_y = x, y

    # ----------------------
    # Krawędzie: Robertsa, Sobel
    # ----------------------
    def show_edge_dialog(self):
        if self.current_image_array is None:
            return
        top = tk.Toplevel(self.root)
        top.title("Wybierz operator krawędzi")

        def roberts():
            self.apply_roberts()
            top.destroy()

        def sobel():
            self.apply_sobel()
            top.destroy()

        tk.Button(top, text="Krzyż Robertsa", command=roberts).pack(pady=5, padx=10)
        tk.Button(top, text="Sobel", command=sobel).pack(pady=5, padx=10)

    def apply_roberts(self):
        # Najczęściej działa na obrazie w skali szarości
        gray = self.to_grayscale(self.current_image_array).astype(np.float32)
        gx_kernel = np.array([[1, 0],
                              [0,-1]], dtype=np.float32)
        gy_kernel = np.array([[0, 1],
                              [-1,0]], dtype=np.float32)

        gx = self.convolve2d(gray, gx_kernel)
        gy = self.convolve2d(gray, gy_kernel)
        grad = np.sqrt(gx*gx + gy*gy)
        grad = np.clip(grad, 0, 255).astype(np.uint8)

        # Zwróćmy wynik w RGB, żeby się poprawnie wyświetlał:
        self.current_image_array = np.stack((grad, grad, grad), axis=-1)
        self.update_display()

    def apply_sobel(self):
        gray = self.to_grayscale(self.current_image_array).astype(np.float32)
        gx_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float32)
        gy_kernel = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]], dtype=np.float32)

        gx = self.convolve2d(gray, gx_kernel)
        gy = self.convolve2d(gray, gy_kernel)
        grad = np.sqrt(gx*gx + gy*gy)
        grad = np.clip(grad, 0, 255).astype(np.uint8)

        self.current_image_array = np.stack((grad, grad, grad), axis=-1)
        self.update_display()

# -------------
# Uruchomienie:
# -------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
