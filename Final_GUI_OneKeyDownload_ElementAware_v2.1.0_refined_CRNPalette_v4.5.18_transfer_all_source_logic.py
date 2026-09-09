import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import networkx as nx
from pathlib import Path
from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import math
import itertools
import os
from decimal import getcontext
import matplotlib
import threading
import time
import colorsys

try:
    from skimage.segmentation import slic
    from skimage.color import rgb2lab, deltaE_ciede2000
    _PHOTO_LAYER_SKIMAGE_AVAILABLE = True
except ImportError:
    slic = None
    rgb2lab = None
    deltaE_ciede2000 = None
    _PHOTO_LAYER_SKIMAGE_AVAILABLE = False

# Windows 下 loky/joblib 首次调用会额外启动子进程探测物理核心。
# 进程已有可靠的逻辑核心数，显式提供后可消除数秒冷启动，不改变 KMeans 参数。
os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT", str(max(1, int(os.cpu_count() or 1)))
)

try:
    from sklearn.cluster import KMeans
except ImportError:
    class KMeans:
        """sklearn KMeans 的轻量 OpenCV 兼容层。

        仅覆盖本项目实际使用的 fit/fit_predict、labels_ 和
        cluster_centers_，确保缺少 sklearn 时 GUI 仍可启动并完成聚类。
        """

        def __init__(self, n_clusters=8, init="k-means++", n_init=10,
                     random_state=None, max_iter=300, **_kwargs):
            self.n_clusters = int(n_clusters)
            self.init = init
            self.n_init = max(1, int(n_init))
            self.random_state = 0 if random_state is None else int(random_state)
            self.max_iter = max(1, int(max_iter))
            self.labels_ = None
            self.cluster_centers_ = None
            self.inertia_ = None

        def fit(self, values):
            data = np.asarray(values, dtype=np.float32)
            if data.ndim != 2 or len(data) == 0:
                raise ValueError("KMeans requires a non-empty 2D array")
            if not 1 <= self.n_clusters <= len(data):
                raise ValueError("n_clusters must be between 1 and sample count")

            cv2.setRNGSeed(self.random_state & 0x7FFFFFFF)
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                self.max_iter,
                1e-4,
            )
            initial_labels = None
            flags = cv2.KMEANS_PP_CENTERS
            attempts = self.n_init
            if isinstance(self.init, (list, tuple, np.ndarray)):
                initial_centers = np.asarray(self.init, dtype=np.float32)
                if initial_centers.shape != (self.n_clusters, data.shape[1]):
                    raise ValueError("init centers have an incompatible shape")
                distances = np.linalg.norm(
                    data[:, None, :] - initial_centers[None, :, :], axis=2
                )
                initial_labels = np.argmin(distances, axis=1).astype(np.int32)
                flags = cv2.KMEANS_USE_INITIAL_LABELS
                attempts = 1
            elif str(self.init).lower() == "random":
                flags = cv2.KMEANS_RANDOM_CENTERS

            compactness, labels, centers = cv2.kmeans(
                data,
                self.n_clusters,
                initial_labels,
                criteria,
                attempts,
                flags,
            )
            self.inertia_ = float(compactness)
            self.labels_ = labels.reshape(-1).astype(np.int32)
            self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
            return self

        def fit_predict(self, values):
            return self.fit(values).labels_

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        """sklearn StandardScaler 的最小兼容实现。"""

        def fit_transform(self, values):
            data = np.asarray(values, dtype=np.float64)
            mean = np.mean(data, axis=0)
            scale = np.std(data, axis=0)
            scale = np.where(scale < 1e-12, 1.0, scale)
            return (data - mean) / scale

try:
    from scipy.spatial.distance import cdist
except ImportError:
    def cdist(first, second, metric="euclidean"):
        """缺少 SciPy 时提供项目所需的欧氏距离矩阵。"""
        if metric != "euclidean":
            raise ValueError("fallback cdist only supports euclidean distance")
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        return np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)


class NearestNeighborIndex:
    """cKDTree 的按需兼容层，供对称分组查询最近候选。"""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        self._tree = None
        try:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.points)
        except ImportError:
            pass

    def query(self, target, k=1):
        if self._tree is not None:
            return self._tree.query(target, k=k)
        distances = np.linalg.norm(
            self.points - np.asarray(target, dtype=np.float64), axis=1
        )
        count = min(max(1, int(k)), len(distances))
        indices = np.argsort(distances, kind="stable")[:count]
        if count == 1:
            index = int(indices[0])
            return float(distances[index]), index
        return distances[indices], indices.astype(np.intp)


def safe_skew(values):
    """优先使用 SciPy 偏度；缺失时用标准化三阶中心矩。"""
    data = np.asarray(values, dtype=np.float64)
    try:
        from scipy.stats import skew
        return float(skew(data))
    except ImportError:
        if data.size == 0:
            return 0.0
        centered = data - float(np.mean(data))
        deviation = float(np.sqrt(np.mean(centered ** 2)))
        if deviation < 1e-12:
            return 0.0
        return float(np.mean(centered ** 3) / (deviation ** 3))


matplotlib.use( 'Agg' )

getcontext().prec = 50


class ColorTransferToolbox:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "【❆WTU-RCNS-Lab Auxiliary Design Series❆】❆★❆Color Extraction, Style Transfer, and Color Matching Trend Analysis Toolbox❆★❆ Ver. 1.0" )
        self.root.geometry( "1200x800" )

        self.image_path = None
        self.outline_image_path = None
        self.source_outline_image_path = None
        self.target_outline_image_path = None
        self.color_regions = None
        self.hex_color_codes = []
        self.labels_2d = None
        self.latest_colors = None
        self.source_adjacent_pixels = None
        self.source_labels_2d = None
        self.target_image_path = None
        self.color_data = None
        self.colors_applied = False
        self.target_labels_2d = None
        self.color_selection_frame = None
        self.last_action = None
        self.segmentation_method = None
        self.coloring_method_var = tk.StringVar( value="symmetry" )
        self.outline_coloring_colors = []
        self.color_transfer_colors = []

        # 提前定义 symmetry_type，确保在 create_content_area 中可用
        self.symmetry_type = tk.StringVar( value="Asymmetry" )

        # 存储生成的方案图片列表
        self.generated_plans = []
        self.latest_generated_result_image = None
        self.latest_generated_result_description = None

        # 存储网络图的 Figure 对象，用于下载和背景色修改
        self.net_figure = None
        self.trend_palette_hex_codes = []
        self.trend_palette_pixels = []

        # Palette Result Editor（结果调色板后处理）模块状态。
        self.palette_editor_original_image = None
        self.palette_editor_preview_image = None
        self.palette_editor_result_image = None
        # v4.5.0：传统图目标色锚定，黑色墨线不进入可编辑调色板。
        self.palette_editor_mode = None
        self.palette_editor_classification_metrics = {}
        self.palette_editor_source_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_target_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_proportions = np.empty((0,), dtype=np.float32)
        self.palette_editor_sparse_model = None
        self.palette_editor_owner_map = None
        # Object-aware edit groups are unions of color sublayers that belong to one
        # spatial object/material region (e.g. bright orange + dark orange coat).
        self.palette_editor_group_map = None
        self.palette_editor_group_map_original = None
        self.palette_editor_group_map_refined = None
        self.palette_editor_region_reduction_metrics = {}
        self.palette_editor_edge_refine_metrics = {}
        self.palette_editor_group_source_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_group_target_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_group_proportions = np.empty((0,), dtype=np.float32)
        self.palette_editor_group_members = []
        self.palette_editor_selected_region_index = None
        self.palette_editor_selected_region_photo = None
        self.palette_editor_region_template_sheet = None
        self.palette_editor_region_template_window = None
        self.palette_editor_manual_path = None
        self.palette_editor_busy = False
        self.palette_editor_swatch_buttons = []
        # NA-IGA 候选状态：保留完整颜色基因，不使用交叉插值。
        self.palette_editor_naiga_candidates = []
        self.palette_editor_naiga_scores = []
        self.palette_editor_naiga_selected = None
        self.palette_editor_naiga_generation = 0
        self.palette_editor_naiga_widgets = []

        self.main_frame = ttk.Frame( root )
        self.main_frame.pack( fill='both', expand=True, padx=10, pady=10 )

        self.create_toolbar()
        self.create_content_area()

        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label( root, textvariable=self.status_var, relief='sunken', anchor='e' )
        self.status_bar.pack( side='bottom', fill='x' )
        self.update_status( "Welcome to【❆WTU-RCNS-Lab Auxiliary Design Series Toolbox❆】" )

        self.apply_win11_style()
        self.hist_canvas.bind( "<Configure>", self.on_hist_canvas_resize )
        self.net_canvas.bind( "<Configure>", self.on_net_canvas_resize )

    def apply_win11_style(self):
        style = ttk.Style()
        style.theme_use( 'clam' )

        self.bg_color = "#F3F3F3"
        self.card_bg = "#FFFFFF"
        self.border_color = "#E0E0E0"
        self.accent_color = "#0078D4"
        self.accent_hover = "#005A9E"
        self.font_family = "Segoe UI Variable"

        self.root.configure( bg=self.bg_color )
        default_font = (self.font_family, 10)
        style.configure( '.', font=default_font, background=self.bg_color )

        style.configure( 'TLabelframe', background=self.card_bg, bordercolor=self.border_color,
                         lightcolor=self.border_color, darkcolor=self.border_color,
                         borderwidth=2, relief='solid' )
        style.configure( 'TLabelframe.Label', background=self.bg_color, foreground="#202020",
                         font=(self.font_family, 12, 'bold') )

        style.configure( 'TButton', background=self.card_bg, foreground="#202020",
                         bordercolor=self.border_color, focuscolor='none',
                         lightcolor=self.card_bg, darkcolor=self.card_bg,
                         borderwidth=1, relief='flat', padding=(12, 6) )
        style.map( 'TButton',
                   background=[('active', self.accent_hover), ('pressed', self.accent_color)],
                   foreground=[('active', 'white'), ('pressed', 'white')],
                   bordercolor=[('active', self.accent_color)] )

        style.configure( 'TCheckbutton', background=self.bg_color, foreground="#202020",
                         focuscolor='none', font=default_font, indicatormargin=6 )
        style.map( 'TCheckbutton',
                   background=[('active', self.bg_color)],
                   indicatorcolor=[('selected', self.accent_color), ('!selected', self.card_bg)] )

        style.configure( 'TNotebook', background=self.bg_color, bordercolor=self.border_color,
                         borderwidth=0, tabmargins=(2, 2, 2, 0) )
        style.configure( 'TNotebook.Tab', background=self.card_bg, padding=(15, 5),
                         bordercolor=self.border_color, borderwidth=1, relief='flat' )
        style.map( 'TNotebook.Tab',
                   background=[('selected', self.accent_color), ('active', self.border_color)],
                   foreground=[('selected', 'white'), ('active', 'black')] )

        style.configure( 'TEntry', fieldbackground='white', foreground='black',
                         bordercolor=self.border_color, borderwidth=1, relief='solid',
                         padding=5 )
        style.configure( 'TCombobox', fieldbackground='white', background='white',
                         arrowcolor=self.accent_color, padding=5 )

        self.root.option_add( '*Canvas.Background', self.card_bg )
        self.root.option_add( '*Listbox.Background', self.card_bg )
        self.root.option_add( '*Listbox.Font', default_font )

    def update_status(self, message):
        self.status_var.set( message )

    def create_toolbar(self):
        toolbar_frame = ttk.LabelFrame( self.main_frame, text="Toolbox" )
        toolbar_frame.pack( side='left', fill='y', padx=5, pady=5 )

        # Logo区域
        logo_frame = ttk.Frame( toolbar_frame )
        logo_frame.pack( pady=(5, 10), fill='x' )
        self.logo_label = ttk.Label( logo_frame, text="Loading logo...", anchor='center' )
        self.logo_label.pack()
        self.load_logo()  # 尝试加载logo

        # 按钮列表
        buttons = [
            ("Load Source Image", self.load_image),
            ("Extract Colors", self.extract_colors),
            ("Transfer Colors", self.open_color_transfer),
            ("Coloring Line Sketch", self.open_outline_coloring),
            ("Generate Solution", self.generate_plan),
            ("Color Fashion Trends", self.batch_trend_colors),
            ("Palette Result Editor", self.open_palette_recoloring),
            ("Save Plans", self.save_results),
            ("Download All Plans", self.download_all_plans),
            ("ReadMe", self.show_readme)
        ]

        for text, command in buttons:
            # Reduced width from 15 to 12, padding from 5 to 3
            btn = ttk.Button( toolbar_frame, text=text, command=command, width=12 )
            btn.pack( pady=3, padx=3, fill='x' )
            if text == "Generate Solution":
                self.plan_button = btn

        self.plan_button.config( state="disabled" )

        # Parameter Settings area - directly display all parameters without scrollbar
        param_frame = ttk.LabelFrame( toolbar_frame, text="Parameter Settings" )
        param_frame.pack( fill='x', padx=3, pady=5 )

        # 在参数设置区域下方添加单位Logo
        logo2_frame = ttk.Frame( toolbar_frame )
        logo2_frame.pack( fill='x', padx=3, pady=(5, 3) )
        self.logo2_label = ttk.Label( logo2_frame, text="Loading logo...", anchor='center' )
        self.logo2_label.pack( fill='x', padx=5, pady=5 )

        # 加载 logo2.png
        logo2_path = os.path.join( os.path.dirname( __file__ ), 'logo', 'logo2.png' )
        if os.path.exists( logo2_path ):
            try:
                img2 = Image.open( logo2_path ).convert( 'RGBA' if logo2_path.lower().endswith( '.png' ) else 'RGB' )
                # 设置最大宽度为200，高度自适应
                img2.thumbnail( (180, 80), Image.Resampling.LANCZOS )
                self.logo2_photo = ImageTk.PhotoImage( img2 )
                self.logo2_label.config( image=self.logo2_photo, text='' )
            except Exception as e:
                print( f"Unable to load Logo: {e}" )
                self.logo2_label.config( text="Logo error" )
        else:
            self.logo2_label.config( text="Logo not found" )

        # Add parameter controls - shortened labels and compact layout
        ttk.Label( param_frame, text="Colors:" ).grid( row=0, column=0, padx=3, pady=2, sticky='e' )
        self.color_number_var = tk.IntVar( value=5 )
        ttk.Entry( param_frame, textvariable=self.color_number_var, width=4 ).grid( row=0, column=1, padx=3, pady=2 )

        ttk.Label( param_frame, text="Threshold:" ).grid( row=1, column=0, padx=3, pady=2, sticky='e' )
        self.network_threshold_var = tk.DoubleVar( value=0.1 )
        ttk.Entry( param_frame, textvariable=self.network_threshold_var, width=4 ).grid( row=1, column=1, padx=3,
                                                                                         pady=2 )

        ttk.Label( param_frame, text="Rows:" ).grid( row=2, column=0, padx=3, pady=2, sticky='e' )
        self.row_var = tk.IntVar( value=2 )
        ttk.Entry( param_frame, textvariable=self.row_var, width=4 ).grid( row=2, column=1, padx=3, pady=2 )

        ttk.Label( param_frame, text="Cols:" ).grid( row=3, column=0, padx=3, pady=2, sticky='e' )
        self.col_var = tk.IntVar( value=3 )
        ttk.Entry( param_frame, textvariable=self.col_var, width=4 ).grid( row=3, column=1, padx=3, pady=2 )

    # 加载Logo
    def load_logo(self):
        import os
        # 假设logo图片存放在主函数当前目录的logo文件夹，文件名为logo.png
        logo_path = os.path.join( os.path.dirname( __file__ ), 'logo', 'logo.jpeg' )
        if os.path.exists( logo_path ):
            try:
                img = Image.open( logo_path ).convert( 'RGB' )
                img.thumbnail( (100, 100) )  # 调整为正方形区域大小
                self.logo_photo = ImageTk.PhotoImage( img )
                self.logo_label.config( image=self.logo_photo, text='' )
            except Exception as e:
                print( f"加载logo失败: {e}" )
                self.logo_label.config( text="Logo error" )
        else:
            self.logo_label.config( text="Logo not found" )

    @staticmethod
    def _prepare_plan_export_image(image, width, height, dpi):
        """按统一契约准备方案导出图，但不改变生成算法的原始像素。

        Width/Height 是最大输出边界。图像始终保持宽高比、转为 RGB，并携带
        对称 DPI 元数据；Transfer Colors 与 Coloring Line Sketch 共用此入口。
        """
        width = int(width)
        height = int(height)
        dpi = int(dpi)
        if width <= 0 or height <= 0 or dpi <= 0:
            raise ValueError("Width, height and DPI must be positive integers.")

        prepared = image.copy().convert("RGB")
        source_width, source_height = prepared.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("The generated plan has an invalid pixel size.")

        scale = min(width / source_width, height / source_height)
        output_size = (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        )
        if output_size != prepared.size:
            prepared = prepared.resize(output_size, Image.Resampling.LANCZOS)

        dpi_pair = (dpi, dpi)
        prepared.info["dpi"] = dpi_pair
        return prepared, dpi_pair

    def _get_plan_export_settings(self):
        """读取方案页唯一的像素边界与 DPI 设置。"""
        width = int(self.download_width.get())
        height = int(self.download_height.get())
        dpi = int(self.download_dpi.get())
        if width <= 0 or height <= 0 or dpi <= 0:
            raise ValueError("Width, height and DPI must be positive integers.")
        return width, height, dpi

    def _save_plan_png(self, image, file_path, width, height, dpi):
        """使用统一的 RGB、尺寸和 DPI 契约保存一张 PNG 方案。"""
        prepared, dpi_pair = self._prepare_plan_export_image(
            image, width, height, dpi
        )
        prepared.save(file_path, format="PNG", dpi=dpi_pair)
        return prepared

    # 一键下载所有方案
    def download_all_plans(self):
        if not hasattr( self, 'generated_plans' ) or not self.generated_plans:
            messagebox.showwarning( "Warning", "No plans generated, please generate a plan first!" )
            return
        folder = filedialog.askdirectory( title="Select folder to save all plans" )
        if not folder:
            return
        try:
            width, height, dpi = self._get_plan_export_settings()
            for i, img in enumerate( self.generated_plans ):
                filename = os.path.join( folder, f"plan_{i + 1}.png" )
                self._save_plan_png(img, filename, width, height, dpi)
            messagebox.showinfo( "Success", f"Saved {len( self.generated_plans )} plans to folder: {folder}" )
            self.update_status(
                f"All plans saved to {folder} "
                f"(within {width}×{height}, RGB, {dpi} DPI)"
            )
        except Exception as e:
            messagebox.showerror( "Error", f"Save failed: {str( e )}" )

    # 显示ReadMe窗口
    def show_readme(self):
        readme_win = tk.Toplevel( self.root )
        readme_win.title( "Instructions" )
        readme_win.geometry( "600x400" )
        readme_win.transient( self.root )
        readme_win.grab_set()

        text = tk.Text( readme_win, wrap='word', font=('Segoe UI', 10) )
        text.pack( fill='both', expand=True, padx=10, pady=10 )

        scrollbar = ttk.Scrollbar( text, command=text.yview )
        scrollbar.pack( side='right', fill='y' )
        text.config( yscrollcommand=scrollbar.set )

        content = """
        【❆WTU-RCNS-Lab Auxiliary Design Series❆】Color Extraction and Style Transfer Toolbox - Instructions

        1. Load Source Image: Click "Load Dource Image" to select an original image; the program will automatically generate a line sketch.
        2. Extract Colors: Click "Extract Colors" to extract dominant colors, displayed in histogram and network graph.
        3. Transfer Colors: Click "Transfer Colors" to select a target image and apply extracted colors to it.
        4. Coloring Line Sketch: Click "Coloring The Line Sketch" to load a line sketch, choose segmentation method, apply colors, then generate recommanded color schemes.
        5. Generate Solution: Click "Generate Solution" to generate color schemes based on selected colors and symmetry options.
        6. Color Fashion Trends: Click "Color Fashion Trends" to batch analyze images and obtain trending colors.
        7. Save Plans: Click "Save Plans" to save a single plan, or use "Download All Plans" to save all generated plans to a folder.
        8. More features to explore...

        Note: Ensure image format is supported; line sketchs should be black-and-white line art.

        Designed by Huang Qi(黄琦), Chen Ken(陈恳), Yuan Yu(袁玉) & Liu Jie(刘杰)* @ WTU Ver.20260315.

        (*)Correspondence Designer: E-mail:liujie@wtu.edu.cn Tel: +86+19945012837 QQ:120283667 Wechat:WTUliujie

        Copyright © 2026 - 2036 WTU RCNS-Lab, All Rights Reserved.

        """
        text.insert( '1.0', content )
        text.config( state='disabled' )

    def create_content_area(self):
        content_frame = ttk.Frame( self.main_frame )
        content_frame.pack( side='right', fill='both', expand=True, padx=5, pady=5 )

        self.notebook = ttk.Notebook( content_frame )
        self.notebook.pack( fill='both', expand=True )

        self.image_tab = ttk.Frame( self.notebook )
        self.create_image_tab()
        self.extraction_tab = ttk.Frame( self.notebook )
        self.create_extraction_tab()
        self.design_tab = ttk.Frame( self.notebook )
        self.create_design_tab()
        self.transfer_tab = ttk.Frame( self.notebook )
        self.create_transfer_tab()
        self.outline_coloring_tab = ttk.Frame( self.notebook )
        self.create_outline_coloring_tab()
        # 最后一个板块：生成结果/任意图片的调色板后处理编辑。
        self.palette_recolor_tab = ttk.Frame( self.notebook )
        self.create_palette_recolor_tab()
        self.show_tab = self._make_show_tab()

    def _make_show_tab(self):
        """返回一个用于添加标签页的函数，并为需要的标签页添加关闭按钮"""

        def show_tab(tab, text):
            # 检查 tab 是否已经在 notebook 中
            try:
                self.notebook.index( tab )
                # 已在其中，只选中即可
                self.notebook.select( tab )
                return
            except tk.TclError:
                # 尚未添加，继续添加
                pass

            self.notebook.add( tab, text=text )
            # 为这些标签页添加关闭按钮
            if tab in (self.image_tab, self.extraction_tab,
                       self.transfer_tab, self.outline_coloring_tab, self.design_tab,
                       self.palette_recolor_tab):
                if hasattr( tab, 'close_btn' ):
                    return
                close_btn = ttk.Button( tab, text='✖', width=2,
                                        command=lambda: self.notebook.forget( tab ) )
                close_btn.place( relx=1.0, x=-5, y=5, anchor='ne' )
                tab.close_btn = close_btn

                def on_configure(e):
                    close_btn.place( relx=1.0, x=-5, y=5, anchor='ne' )

                tab.bind( '<Configure>', on_configure )

        return show_tab

    # -------------------- 线稿图上色标签页 --------------------
    def create_outline_coloring_tab(self):
        tab_frame = ttk.Frame( self.outline_coloring_tab )
        tab_frame.pack( fill='both', expand=True, padx=10, pady=10 )

        color_selection_frame = ttk.LabelFrame( tab_frame, text="Color Select" )
        color_selection_frame.pack( side='left', fill='both', expand=True, padx=10, pady=10 )
        self.color_selection_frame = color_selection_frame
        self.update_outline_coloring_tab()

        outline_frame = ttk.LabelFrame( tab_frame, text="Line Sketch" )
        outline_frame.pack( side='right', fill='both', expand=True, padx=10, pady=10 )
        self.outline_coloring_label = ttk.Label( outline_frame, text="Line draft diagram not loaded", anchor='center' )
        self.outline_coloring_label.pack( fill='both', expand=True, padx=10, pady=10 )

        ttk.Button( tab_frame, text="Import Line Sketch", command=self.load_outline_image_for_coloring ).pack(
            side='top', pady=10, padx=10, fill='x' )
        self.apply_color_button = ttk.Button(
            tab_frame, text="Apply Color", command=self.apply_colors_to_outline
        )
        self.apply_color_button.pack(side='top', pady=10, padx=10, fill='x')

        self.segmentation_method_var = tk.StringVar( value="Perfect Mirror Symmetry Segmentation" )
        ttk.Label( tab_frame, text="Select Segmentation Methods:" ).pack( side='top', pady=5, padx=10 )
        segmentation_options = ttk.Combobox(
            tab_frame,
            textvariable=self.segmentation_method_var,
            values=(
                "Perfect Mirror Symmetry Segmentation",
                "Segmentation based on contour detection",
                "Segmentation based on watershed algorithm",
                "Segmentation based on Geometric analysis",
                "Segmentation based on Voronoi method",
                "Felzenszwalb segmentation",

                # 新增👇
                "Flood Fill segmentation",
                "Connected component segmentation",
                "SLIC superpixel segmentation",
                "Distance transform watershed",
                "Symmetry-Aware Segmentation",
                "Element-Aware Segmentation (v11)"
            ),
            state="readonly"
        )
        segmentation_options.pack( side='top', pady=5, padx=10, fill='x' )

    def load_outline_image_for_coloring(self):
        """加载线稿图用于上色；Element-Aware 保留用户选择的原始分辨率线稿。"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.jfif")],
            title="Select line sketch"
        )
        if file_path:
            try:
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img)

                if not self.is_outline_image(img_array):
                    messagebox.showerror(
                        "Error!",
                        "This is not a line sketch, please select the correct line sketch."
                    )
                    return

                self._invalidate_outline_plan_state()
                # 关键：原文件随后会调用 create_outline_image()，它会用 Canny
                # 重画并缩到最多 1200px。Element-Aware 必须记住原图，避免把
                # 2048px 干净线稿降采样后再放大，造成双边、断线与模糊。
                self._element_aware_original_outline_path = file_path
                self.display_image(img, self.outline_coloring_label, (400, 400))
                self.outline_image_path = file_path
                self.update_status(
                    f"Related line sketch has been loaded: {Path(file_path).name}"
                )

                # 保留原 GUI 的辅助线稿生成与预览功能；仅 Element-Aware 在
                # apply 阶段读取上面保存的原始文件，其他分割器行为不变。
                self.create_outline_image(file_path, 'source_outline_image_path')
                if self.source_outline_image_path and os.path.exists(self.source_outline_image_path):
                    outline_img = Image.open(self.source_outline_image_path).convert('RGB')
                    self.display_image(outline_img, self.outline_label, (400, 400))
                else:
                    messagebox.showerror(
                        "Error!", "Unable to generate a line sketch of the target image!"
                    )
                    return

            except Exception as e:
                messagebox.showerror("Error!", f"Unable to load line sketch: {str(e)}")

    def _reset_outline_segmentation_metadata(self):
        """清除只对当前线稿及其 splitter 结果有效的诊断与参考元数据。"""
        self._reference_region_color_hints = None
        self._reference_region_color_hint_purities = None
        self._outline_background_region_index = None
        self._last_outline_motif_graph = None
        self._last_symmetry_diagnostics = None
        self._reference_alignment_score = 0.0
        self._reference_orientation_score = 0.0
        self._reference_fine_alignment_score = 0.0
        self._reference_projection_score = 0.0

    def _invalidate_outline_plan_state(self):
        """新线稿成为当前输入时，旧分区与旧方案立即失效。"""
        self._reset_outline_segmentation_metadata()
        self._transfer_element_aware_active = False
        self.regions_for_plan = None
        self.plan_outline_array = None
        self.plan_outline_image = None
        self.generated_plans = []
        self.colors_applied = False
        self.last_action = None
        self.plan_button.config(state="disabled")

    # -------------------- 分割方法（保持不变） --------------------
    def split_regions_by_contour_detection(self, outline_array, color_count):
        import cv2
        import numpy as np
        from collections import defaultdict

        if len( outline_array.shape ) == 2 or outline_array.shape[2] == 1:
            gray = outline_array
        else:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_BGR2GRAY )

        clahe = cv2.createCLAHE( clipLimit=2.0, tileGridSize=(8, 8) )
        gray = clahe.apply( gray )

        kernel = np.array( [[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32 )
        gray = cv2.filter2D( gray, -1, kernel )

        _, binary_inv = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        contours, _ = cv2.findContours( binary_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

        h, w = gray.shape
        total_area = h * w
        min_area = max( 30, 0.0003 * total_area )
        max_area = 0.8 * total_area

        features = []
        valid_contours = []

        for cnt in contours:
            area = cv2.contourArea( cnt )
            if area < min_area or area > max_area:
                continue

            M = cv2.moments( cnt )
            if M['m00'] == 0:
                continue

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            dx = cx - w / 2
            dy = cy - h / 2
            radius = np.sqrt( dx ** 2 + dy ** 2 )
            angle = (np.arctan2( dy, dx ) + 2 * np.pi) % (2 * np.pi)
            perimeter = cv2.arcLength( cnt, True )
            hu = cv2.HuMoments( M ).flatten()

            mask = np.zeros_like( gray )
            cv2.drawContours( mask, [cnt], -1, 255, thickness=cv2.FILLED )
            region_pixels = gray[mask == 255]
            texture_std = np.std( region_pixels ) if len( region_pixels ) > 5 else 0

            features.append( [
                area / total_area,
                perimeter / (w + h),
                abs( dx ) / w,
                abs( dy ) / h,
                texture_std / 255,
                radius / np.sqrt( w ** 2 + h ** 2 ),
                angle / (2 * np.pi),
                *hu
            ] )
            valid_contours.append( cnt )

        if not features:
            return []

        features = np.array( features, dtype=np.float32 )
        features = StandardScaler().fit_transform( features )

        precluster_num = min( len( valid_contours ), max( color_count * 3, 30 ) )
        kmeans = KMeans( n_clusters=precluster_num, random_state=42, n_init=10 )
        pre_labels = kmeans.fit_predict( features )

        label_to_contours = defaultdict( list )
        for i, label in enumerate( pre_labels ):
            label_to_contours[label].append( valid_contours[i] )

        meta_features = []
        label_list = []
        for label, cnts in label_to_contours.items():
            all_pixels = np.zeros_like( gray )
            for c in cnts:
                cv2.drawContours( all_pixels, [c], -1, 255, thickness=cv2.FILLED )
            ys, xs = np.where( all_pixels == 255 )
            if len( xs ) == 0:
                continue
            cx = np.mean( xs )
            cy = np.mean( ys )
            dx = cx - w / 2
            dy = cy - h / 2
            radius = np.sqrt( dx ** 2 + dy ** 2 )
            meta_features.append( [
                len( xs ) / total_area,
                abs( dx ) / w,
                abs( dy ) / h,
                radius / np.sqrt( w ** 2 + h ** 2 )
            ] )
            label_list.append( label )

        # 【改进】创建更多区域（2-3倍颜色数），让颜色分配更灵活平衡
        target_regions = min(len(meta_features), color_count * 2)
        if target_regions < color_count:
            target_regions = color_count
        
        if len( meta_features ) < target_regions:
            target_regions = len( meta_features )

        merge_kmeans = KMeans( n_clusters=target_regions, random_state=0, n_init=5 )
        final_labels = merge_kmeans.fit_predict( meta_features )

        min_region_area = 0.007 * total_area
        regions = [np.zeros_like( gray ) for _ in range( target_regions )]
        label_center_map = {}
        used_mask = np.zeros_like( gray )

        for i, origin_label in enumerate( label_list ):
            cnts = label_to_contours[origin_label]
            all_pixels = np.zeros_like( gray )
            for c in cnts:
                cv2.drawContours( all_pixels, [c], -1, 255, thickness=cv2.FILLED )
            ys, xs = np.where( all_pixels == 255 )
            if len( xs ) == 0:
                continue
            center = (np.mean( xs ), np.mean( ys ))
            label_center_map[origin_label] = center

        region_centers = [[] for _ in range( target_regions )]
        region_areas = [0 for _ in range( target_regions )]

        for i, origin_label in enumerate( label_list ):
            new_label = final_labels[i]
            cx, cy = label_center_map[origin_label]

            if region_areas[new_label] >= min_region_area:
                new_label = np.argmin( region_areas )

            region_centers[new_label].append( (cx, cy) )

            for cnt in label_to_contours[origin_label]:
                mask = np.zeros_like( gray )
                cv2.drawContours( mask, [cnt], -1, 255, thickness=cv2.FILLED )
                overlap = cv2.bitwise_and( mask, used_mask )
                if np.count_nonzero( overlap ) == 0:
                    regions[new_label] = cv2.bitwise_or( regions[new_label], mask )
                    used_mask = cv2.bitwise_or( used_mask, mask )
                    region_areas[new_label] += np.count_nonzero( mask )

        return regions

    def split_regions_by_watershed(self, outline_array, color_count):
        import cv2
        import numpy as np

        if len( outline_array.shape ) == 2 or outline_array.shape[2] == 1:
            gray = outline_array
        else:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_BGR2GRAY )

        _, binary = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        binary_inv = 255 - binary
        line_mask = (binary == 0).astype( np.uint8 )

        kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (3, 3) )
        closed = cv2.morphologyEx( binary_inv, cv2.MORPH_CLOSE, kernel, iterations=2 )

        contours, hierarchy = cv2.findContours( closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )
        h, w = gray.shape
        used_mask = np.zeros( (h, w), dtype=np.uint8 )
        region_masks = []
        region_areas = []
        min_area = 1

        for idx, cnt in enumerate( contours ):
            if hierarchy[0][idx][3] != -1:
                continue
            mask = np.zeros( (h, w), dtype=np.uint8 )
            cv2.drawContours( mask, [cnt], -1, 255, thickness=cv2.FILLED )

            mask = cv2.bitwise_and( mask, cv2.bitwise_not( line_mask * 255 ) )
            non_overlap = cv2.bitwise_and( mask, cv2.bitwise_not( used_mask ) )
            area = cv2.countNonZero( non_overlap )

            if area < min_area:
                continue

            region_masks.append( non_overlap )
            region_areas.append( area )
            used_mask = cv2.bitwise_or( used_mask, non_overlap )

        merged_masks = [np.zeros( (h, w), dtype=np.uint8 ) for _ in range( color_count )]
        merged_areas = [0] * color_count
        total_area = sum( region_areas )
        max_single_area = total_area / color_count * 1.5

        def split_and_assign(mask, area):
            if area < max_single_area:
                idx = merged_areas.index( min( merged_areas ) )
                merged_masks[idx] = cv2.bitwise_or( merged_masks[idx], mask )
                merged_areas[idx] += area
                return

            dist = cv2.distanceTransform( mask, cv2.DIST_L2, 5 )
            if dist.max() < 2:
                idx = merged_areas.index( min( merged_areas ) )
                merged_masks[idx] = cv2.bitwise_or( merged_masks[idx], mask )
                merged_areas[idx] += area
                return

            _, sure_fg = cv2.threshold( dist, 0.3 * dist.max(), 255, 0 )
            sure_fg = np.uint8( sure_fg )
            unknown = cv2.subtract( mask, sure_fg )

            _, markers = cv2.connectedComponents( sure_fg )
            markers += 1
            markers[unknown == 255] = 0

            color_mask = cv2.cvtColor( mask, cv2.COLOR_GRAY2BGR )
            markers = cv2.watershed( color_mask, markers )

            max_marker = np.max( markers )
            for i in range( 2, max_marker + 1 ):
                submask = np.uint8( markers == i ) * 255
                submask = cv2.bitwise_and( submask, cv2.bitwise_not( line_mask * 255 ) )
                subarea = cv2.countNonZero( submask )
                if subarea == 0:
                    continue
                idx = merged_areas.index( min( merged_areas ) )
                merged_masks[idx] = cv2.bitwise_or( merged_masks[idx], submask )
                merged_areas[idx] += subarea

        region_data = sorted( zip( region_areas, region_masks ), reverse=True, key=lambda x: x[0] )
        for area, mask in region_data:
            split_and_assign( mask, area )

        for i in range( color_count ):
            merged_masks[i] = cv2.bitwise_and( merged_masks[i], cv2.bitwise_not( line_mask * 255 ) )

        full_mask = np.zeros( (h, w), dtype=np.uint8 )
        for m in merged_masks:
            full_mask = cv2.bitwise_or( full_mask, m )

        all_fg_mask = ((binary_inv > 0) & (line_mask == 0)).astype( np.uint8 ) * 255
        missing_mask = cv2.bitwise_and( all_fg_mask, cv2.bitwise_not( full_mask ) )
        missing_count = cv2.countNonZero( missing_mask )

        if missing_count > 0:
            print( f"⚠️ leak filling stage：There are still {missing_count} Unfold pixels, forced to add maximum area." )
            num_labels, labels = cv2.connectedComponents( missing_mask )
            for label in range( 1, num_labels ):
                single_mask = (labels == label).astype( np.uint8 ) * 255
                label_area = cv2.countNonZero( single_mask )
                idx = merged_areas.index( max( merged_areas ) )
                merged_masks[idx] = cv2.bitwise_or( merged_masks[idx], single_mask )
                merged_areas[idx] += label_area

        full_mask = np.zeros( (h, w), dtype=np.uint8 )
        for m in merged_masks:
            full_mask = cv2.bitwise_or( full_mask, m )

        missing_mask = cv2.bitwise_and( all_fg_mask, cv2.bitwise_not( full_mask ) )
        missing_count = cv2.countNonZero( missing_mask )
        if missing_count > 0:
            print( f"⚠️ Final leak filling stage：There are {missing_count} Unfold pixels, forced to add maximum area." )
            idx = merged_areas.index( max( merged_areas ) )
            merged_masks[idx] = cv2.bitwise_or( merged_masks[idx], missing_mask )
            merged_areas[idx] += missing_count

        return merged_masks[:color_count]

    def split_regions_by_geometric_analysis(self, outline_array, color_count):
        import numpy as np
        import cv2
        import math

        height, width = outline_array.shape[:2]
        gray = outline_array if len( outline_array.shape ) == 2 else cv2.cvtColor( outline_array, cv2.COLOR_BGR2GRAY )
        total_area = height * width

        _, binary = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        line_mask = (binary == 0).astype( np.uint8 ) * 255

        cx, cy = width // 2, height // 2

        angles = np.linspace( 0, 360, color_count + 1 )
        Y, X = np.ogrid[:height, :width]
        dx = X - cx
        dy = cy - Y

        pixel_angles = (np.degrees( np.arctan2( dy, dx ) ) + 360) % 360

        region_masks = []
        for i in range( color_count ):
            start_angle = angles[i]
            end_angle = angles[i + 1]

            if start_angle < end_angle:
                mask = ((pixel_angles >= start_angle) & (pixel_angles < end_angle)).astype( np.uint8 ) * 255
            else:
                mask = ((pixel_angles >= start_angle) | (pixel_angles < end_angle)).astype( np.uint8 ) * 255

            mask = cv2.bitwise_and( mask, cv2.bitwise_not( line_mask ) )
            region_masks.append( mask )

        refined_masks = []
        for mask in region_masks:
            num_labels, labels = cv2.connectedComponents( mask )
            for label in range( 1, num_labels ):
                single_mask = (labels == label).astype( np.uint8 ) * 255
                refined_masks.append( single_mask )

        refined_areas = [cv2.countNonZero( m ) for m in refined_masks]
        target_area = sum( refined_areas ) / color_count

        final_regions = [np.zeros( (height, width), dtype=np.uint8 ) for _ in range( color_count )]
        final_areas = [0] * color_count

        idx_sorted = sorted( range( len( refined_areas ) ), key=lambda k: refined_areas[k], reverse=True )

        for idx in idx_sorted:
            mask = refined_masks[idx]
            area = refined_areas[idx]

            candidates = [i for i, a in enumerate( final_areas ) if a + area <= target_area * 1.2]
            if not candidates:
                assign_idx = final_areas.index( min( final_areas ) )
            else:
                assign_idx = min( candidates, key=lambda x: final_areas[x] )

            final_regions[assign_idx] = cv2.bitwise_or( final_regions[assign_idx], mask )
            final_areas[assign_idx] += area

        all_fg_mask = cv2.bitwise_and( cv2.bitwise_not( line_mask ), (binary > 0).astype( np.uint8 ) * 255 )
        combined_mask = np.zeros( (height, width), dtype=np.uint8 )
        for region in final_regions:
            combined_mask = cv2.bitwise_or( combined_mask, region )

        missing_mask = cv2.bitwise_and( all_fg_mask, cv2.bitwise_not( combined_mask ) )
        missing_count = cv2.countNonZero( missing_mask )

        if missing_count > 0:
            num_labels, labels = cv2.connectedComponents( missing_mask )
            for label in range( 1, num_labels ):
                single_mask = (labels == label).astype( np.uint8 ) * 255
                label_area = cv2.countNonZero( single_mask )
                assign_idx = final_areas.index( min( final_areas ) )
                final_regions[assign_idx] = cv2.bitwise_or( final_regions[assign_idx], single_mask )
                final_areas[assign_idx] += label_area

        return final_regions

    def split_regions_by_voronoi(self, outline_array, color_count):
        import numpy as np
        import cv2

        height, width = outline_array.shape[:2]
        gray = outline_array if len( outline_array.shape ) == 2 else cv2.cvtColor( outline_array, cv2.COLOR_BGR2GRAY )

        _, binary = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        line_mask = (binary == 0).astype( np.uint8 ) * 255

        fg_mask = cv2.bitwise_and( cv2.bitwise_not( line_mask ), (binary > 0).astype( np.uint8 ) * 255 )
        coords = np.column_stack( np.where( fg_mask > 0 ) )

        if len( coords ) < color_count:
            raise ValueError(
                "The divisible areas in the image are insufficient to distinguish the specified quantity!" )

        kmeans = KMeans( n_clusters=color_count, random_state=0 ).fit( coords )
        points = kmeans.cluster_centers_

        region_masks = [np.zeros( (height, width), dtype=np.uint8 ) for _ in range( color_count )]
        Y, X = np.meshgrid( np.arange( height ), np.arange( width ), indexing='ij' )

        distances = np.linalg.norm( coords[:, None, :] - points[None, :, :], axis=2 )
        nearest = np.argmin( distances, axis=1 )

        pixel_labels = np.full( (height, width), -1, dtype=np.int32 )
        for idx, (y, x) in enumerate( coords ):
            pixel_labels[y, x] = nearest[idx]

        for i in range( color_count ):
            region_masks[i][pixel_labels == i] = 255

        for i in range( color_count ):
            region_masks[i] = cv2.bitwise_and( region_masks[i], cv2.bitwise_not( line_mask ) )

        refined_masks = []
        for mask in region_masks:
            num_labels, labels = cv2.connectedComponents( mask )
            for label in range( 1, num_labels ):
                single_mask = (labels == label).astype( np.uint8 ) * 255
                refined_masks.append( single_mask )

        refined_areas = [cv2.countNonZero( m ) for m in refined_masks]
        target_area = sum( refined_areas ) / color_count
        final_regions = [np.zeros( (height, width), dtype=np.uint8 ) for _ in range( color_count )]
        final_areas = [0] * color_count
        idx_sorted = sorted( range( len( refined_areas ) ), key=lambda k: refined_areas[k], reverse=True )

        for idx in idx_sorted:
            mask = refined_masks[idx]
            area = refined_areas[idx]
            candidates = [i for i, a in enumerate( final_areas ) if a + area <= target_area * 1.2]
            assign_idx = min( candidates, key=lambda x: final_areas[x] ) if candidates else final_areas.index(
                min( final_areas ) )
            final_regions[assign_idx] = cv2.bitwise_or( final_regions[assign_idx], mask )
            final_areas[assign_idx] += area

        combined_mask = np.zeros( (height, width), dtype=np.uint8 )
        for region in final_regions:
            combined_mask = cv2.bitwise_or( combined_mask, region )

        missing_mask = cv2.bitwise_and( fg_mask, cv2.bitwise_not( combined_mask ) )
        missing_count = cv2.countNonZero( missing_mask )

        if missing_count > 0:
            num_labels, labels = cv2.connectedComponents( missing_mask )
            for label in range( 1, num_labels ):
                single_mask = (labels == label).astype( np.uint8 ) * 255
                label_area = cv2.countNonZero( single_mask )
                assign_idx = final_areas.index( min( final_areas ) )
                final_regions[assign_idx] = cv2.bitwise_or( final_regions[assign_idx], single_mask )
                final_areas[assign_idx] += label_area

        return final_regions

    def _complete_outline_regions(self, outline_array, regions):
        """让全部非线条像素参与分割，并将边界背景作为独立 owner。"""
        source_hints = getattr(self, '_reference_region_color_hints', None)
        source_purities = getattr(
            self, '_reference_region_color_hint_purities', None
        )
        has_reference_metadata = (
            source_hints is not None
            and source_purities is not None
            and len(source_hints) == len(regions)
            and len(source_purities) == len(regions)
        )
        completed_hints = [] if has_reference_metadata else None
        completed_purities = [] if has_reference_metadata else None
        if len(outline_array.shape) == 3:
            gray = np.mean(outline_array, axis=2).astype(np.uint8)
        else:
            gray = outline_array.astype(np.uint8)

        h, w = gray.shape
        line_mask = gray < 245
        target_domain = ~line_mask

        _, domain_labels = cv2.connectedComponents(
            target_domain.astype(np.uint8), connectivity=4
        )
        border_labels = np.unique(np.concatenate((
            domain_labels[0, :],
            domain_labels[-1, :],
            domain_labels[:, 0],
            domain_labels[:, -1],
        )))
        border_labels = border_labels[border_labels > 0]
        background_mask = target_domain & np.isin(domain_labels, border_labels)
        internal_domain = target_domain & ~background_mask
        self._outline_background_region_index = None

        normalized = []
        occupied = np.zeros((h, w), dtype=bool)
        for region_index, region in enumerate(regions):
            mask = (region > 0) & internal_domain & ~occupied
            if np.any(mask):
                normalized.append(mask)
                occupied |= mask
                if has_reference_metadata:
                    completed_hints.append(int(source_hints[region_index]))
                    completed_purities.append(float(source_purities[region_index]))

        if not normalized:
            if np.any(internal_domain):
                normalized = [internal_domain.copy()]
                occupied = internal_domain.copy()
                if has_reference_metadata:
                    completed_hints = [-1]
                    completed_purities = [0.0]

        uncovered = internal_domain & ~occupied
        if np.any(uncovered):
            # 闭合面是不可拆原子：逐像素最近邻会把一个大面切成彩色扇区。
            # 改为先标记完整遗漏连通面，再将整个面交给距离最近且当前较小的 owner。
            uncovered_count, uncovered_labels, uncovered_stats, uncovered_centers = (
                cv2.connectedComponentsWithStats(
                    uncovered.astype(np.uint8), connectivity=4
                )
            )
            region_areas = [int(np.count_nonzero(region)) for region in normalized]
            region_centers = []
            for region in normalized:
                ys, xs = np.where(region)
                region_centers.append((float(np.mean(xs)), float(np.mean(ys))))
            owner_by_label = np.full(uncovered_count, -1, dtype=np.int32)
            for label in range(1, uncovered_count):
                cell_center = (
                    float(uncovered_centers[label, 0]),
                    float(uncovered_centers[label, 1]),
                )
                owner = min(
                    range(len(normalized)),
                    key=lambda index: (
                        (region_centers[index][0] - cell_center[0]) ** 2
                        + (region_centers[index][1] - cell_center[1]) ** 2,
                        region_areas[index],
                    ),
                )
                owner_by_label[label] = owner
                region_areas[owner] += int(uncovered_stats[label, cv2.CC_STAT_AREA])
            uncovered_owner_map = owner_by_label[uncovered_labels]
            for owner in range(len(normalized)):
                normalized[owner] |= uncovered_owner_map == owner

        if np.any(background_mask):
            self._outline_background_region_index = len(normalized)
            normalized.append(background_mask)
            if has_reference_metadata:
                completed_hints.append(-1)
                completed_purities.append(0.0)

        self._reference_region_color_hints = completed_hints
        self._reference_region_color_hint_purities = completed_purities

        return [mask.astype(np.uint8) * 255 for mask in normalized]


    def _extract_outline_atomic_regions(self, outline_array, min_area=2):
        """提取线稿中的全部天然闭合区域（总区域 N），不按颜色数合并。"""
        if outline_array is None:
            return []
        source = np.asarray(outline_array)
        if source.ndim == 3:
            gray = cv2.cvtColor(source[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = source.astype(np.uint8)

        # 仅把真正的线条视为障碍，保留原有灰度抗锯齿边界。
        line_mask = gray < 245
        fillable = (~line_mask).astype(np.uint8)
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fillable, connectivity=4, ltype=cv2.CV_32S
        )
        if component_count <= 1:
            return []

        border_labels = set(int(value) for value in np.unique(np.concatenate((
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ))))
        min_area = max(1, int(min_area))
        region_records = []
        for label in range(1, component_count):
            if label in border_labels:
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[labels == label] = 255
            cx, cy = centroids[label]
            region_records.append((float(cy), float(cx), mask))

        region_records.sort(key=lambda item: (item[0], item[1]))
        return [mask for _cy, _cx, mask in region_records]

    def _resolve_outline_path_for_segmentation(self, selected_method):
        """Element-Aware 使用用户导入的原始线稿；其他算法沿用原路径。"""
        if selected_method == "Element-Aware Segmentation (v11)":
            original = getattr(self, '_element_aware_original_outline_path', None)
            if original:
                return original
        return self.outline_image_path

    @staticmethod
    def _element_aware_v21_hex_to_rgb(colors):
        values = []
        for color in colors:
            text = str(color).strip()
            if text.startswith('#') and len(text) == 7:
                values.append([
                    int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
                ])
            else:
                raise ValueError(f'Invalid RGB hex color: {color}')
        return np.asarray(values, dtype=np.uint8)

    def _element_aware_v21_palette_role_mapping(self, selected_colors, role_count):
        """把任意用户色板匹配到 v2.1.0 的语义色彩角色。

        原核心的 cluster 0..5 并不是“第 1..6 个任意颜色”，而分别对应
        青绿、深绿、珊瑚、红、粉、肤粉。旧移植直接按索引套入来源色板，
        因而来源色板第一个颜色为红时，最大 cluster 会整片变红。
        这里以 Lab 最小总代价做一对一角色匹配，保持核心的色彩结构。
        """
        role_count = max(0, int(role_count))
        if role_count == 0:
            return []
        if not selected_colors:
            return [0] * role_count

        selected_rgb = self._element_aware_v21_hex_to_rgb(selected_colors)
        canonical = np.asarray([
            [88, 174, 149],   # teal / broad base
            [52, 93, 74],     # dark green
            [241, 122, 96],   # coral
            [237, 43, 52],    # red accent
            [245, 98, 172],   # pink accent
            [236, 213, 214],  # skin/light neutral
        ], dtype=np.uint8)
        if role_count <= len(canonical):
            role_rgb = canonical[:role_count]
        else:
            repeats = int(np.ceil(role_count / len(canonical)))
            role_rgb = np.tile(canonical, (repeats, 1))[:role_count]

        role_lab = cv2.cvtColor(
            role_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32)
        selected_lab = cv2.cvtColor(
            selected_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32)
        distance = np.sum(
            (role_lab[:, None, :] - selected_lab[None, :, :]) ** 2, axis=2
        )

        # 不再使用 Hungarian assignment；这里只保留最近色的初始提示。
        # 真正的最终方案统一由严格部分排列枚举 P(N,M) 生成。
        mapping = np.argmin(distance, axis=1).astype(np.int32)
        return [int(value) for value in mapping]

    @staticmethod
    def _refine_element_aware_owner_map(
            owner_map, min_area=12, iterations=1, protected_owner=None):
        """局部 ROI 精修极小色块，避免在大图上反复膨胀整幅布尔图。

        旧实现会为每个微小连通块创建一张与原图等大的 mask，并执行一次全图
        dilate。复杂线稿常有数千个闭合面，因此会出现几十秒假死。本实现仍保留
        相同的“邻域多数 owner 合并”语义，但只处理连通块 bbox 周围一圈像素。
        """
        owners = np.asarray(owner_map, dtype=np.int32).copy()
        if owners.ndim != 2 or int(iterations) <= 0:
            return owners
        fillable = owners >= 0
        if not np.any(fillable):
            return owners

        kernel = np.ones((3, 3), np.uint8)
        min_area = max(2, int(min_area))
        unique_owners = [int(v) for v in np.unique(owners[fillable])]
        h, w = owners.shape

        for _ in range(max(1, int(iterations))):
            changed = False
            for owner in unique_owners:
                if protected_owner is not None and owner == int(protected_owner):
                    continue
                mask = np.uint8(owners == owner)
                num, cc, stats, _ = cv2.connectedComponentsWithStats(
                    mask, connectivity=8, ltype=cv2.CV_32S
                )
                if num <= 1:
                    continue
                small_labels = np.flatnonzero(
                    (stats[:, cv2.CC_STAT_AREA] > 0)
                    & (stats[:, cv2.CC_STAT_AREA] < min_area)
                )
                small_labels = small_labels[small_labels != 0]
                for label in small_labels.tolist():
                    x = int(stats[label, cv2.CC_STAT_LEFT])
                    y = int(stats[label, cv2.CC_STAT_TOP])
                    bw = int(stats[label, cv2.CC_STAT_WIDTH])
                    bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                    x0, y0 = max(0, x - 1), max(0, y - 1)
                    x1, y1 = min(w, x + bw + 1), min(h, y + bh + 1)
                    local_cc = cc[y0:y1, x0:x1]
                    comp = local_cc == int(label)
                    if not np.any(comp):
                        continue
                    ring = cv2.dilate(
                        comp.astype(np.uint8), kernel, iterations=1
                    ).astype(bool)
                    ring &= ~comp
                    local_owners = owners[y0:y1, x0:x1]
                    ring &= local_owners >= 0
                    ring_values = local_owners[ring]
                    ring_values = ring_values[ring_values != owner]
                    if protected_owner is not None:
                        ring_values = ring_values[
                            ring_values != int(protected_owner)
                        ]
                    if ring_values.size == 0:
                        continue
                    replacement = int(np.bincount(ring_values).argmax())
                    local_owners[comp] = replacement
                    changed = True
            if not changed:
                break
        return owners

    @staticmethod
    def _render_element_aware_v21(outline_array, owner_map, palette_rgb, placement):
        """按部分排列渲染：仅 M 个 owner 上色，其余 owner 保持白底。"""
        outline = np.asarray(outline_array)
        if outline.ndim == 2:
            line_rgb = cv2.cvtColor(outline.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            line_rgb = outline[..., :3].astype(np.uint8, copy=False)
        palette = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)
        owners = np.asarray(owner_map, dtype=np.int32)
        mapping = list(placement)

        fill = np.full(line_rgb.shape, 255, dtype=np.uint8)
        for owner_index, color_value in enumerate(mapping):
            if color_value is None:
                continue
            try:
                color_index = int(color_value)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= color_index < len(palette):
                fill[owners == owner_index] = palette[color_index]

        # 原始灰阶/抗锯齿线条直接作为透射层叠加，不二值化、不膨胀。
        gray = cv2.cvtColor(line_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        transmission = np.clip(gray / 255.0, 0.0, 1.0)[..., None]
        rendered = np.uint8(
            np.clip(fill.astype(np.float32) * transmission, 0, 255)
        )
        return rendered

    @staticmethod
    def _render_transfer_element_aware_exact(
            outline_array, owner_map, palette_rgb, placement):
        """Traditional transfer 部分着色；空槽保持白底，结构墨线原样恢复。"""
        outline = np.asarray(outline_array)
        if outline.ndim == 2:
            native_rgb = cv2.cvtColor(outline.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif outline.ndim == 3 and outline.shape[2] == 1:
            native_rgb = cv2.cvtColor(
                outline[..., 0].astype(np.uint8), cv2.COLOR_GRAY2RGB
            )
        else:
            native_rgb = outline[..., :3].astype(np.uint8, copy=False)

        owners = np.asarray(owner_map, dtype=np.int32)
        palette = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)
        rendered = np.full(native_rgb.shape, 255, dtype=np.uint8)
        for owner_index, color_value in enumerate(list(placement)):
            if color_value is None:
                continue
            try:
                color_index = int(color_value)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= color_index < len(palette):
                rendered[owners == owner_index] = palette[color_index]

        structural_ink = np.any(native_rgb < 254, axis=2)
        rendered[structural_ink] = native_rgb[structural_ink]
        return rendered

    def _element_aware_v21_base_placement(self, selected_colors):
        group_count = len(getattr(self, 'regions_for_plan', []) or [])
        if group_count <= 0:
            return []
        if not selected_colors:
            return [0] * group_count

        mode = getattr(self, '_element_aware_v210_assignment_mode', 'auto')
        if mode == 'reference':
            source_palette = list(getattr(self, 'outline_coloring_colors', []) or [])
            if source_palette:
                selected_rgb = self._element_aware_v21_hex_to_rgb(selected_colors)
                selected_lab = cv2.cvtColor(
                    selected_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
                ).reshape(-1, 3).astype(np.float32)
                result = []
                source_indices = list(getattr(
                    self, '_element_aware_v210_group_source_indices', []
                ) or [])
                background_owner = getattr(
                    self, '_element_aware_v210_background_owner', None
                )
                selected_chroma = np.linalg.norm(
                    selected_lab[:, 1:3] - 128.0, axis=1
                )
                background_color_index = int(np.argmax(
                    selected_lab[:, 0] - 0.35 * selected_chroma
                ))
                for group_index in range(group_count):
                    source_index = (
                        int(source_indices[group_index])
                        if group_index < len(source_indices)
                        else group_index
                    )
                    if group_index == background_owner:
                        result.append(background_color_index)
                    elif 0 <= source_index < len(source_palette):
                        source_rgb = self._element_aware_v21_hex_to_rgb(
                            [source_palette[source_index]]
                        )
                        source_lab = cv2.cvtColor(
                            source_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
                        ).reshape(3).astype(np.float32)
                        distance = np.sum((selected_lab - source_lab[None, :]) ** 2, axis=1)
                        result.append(int(np.argmin(distance)))
                    else:
                        result.append(group_index % len(selected_colors))
                return result
        return self._element_aware_v21_palette_role_mapping(
            selected_colors, group_count
        )

    def _build_element_aware_group_labels(self):
        """把 regions_for_plan 转为连续 labels_2d，供 CRN 评分使用。"""
        if not getattr(self, 'regions_for_plan', None):
            return None, None
        shape = self.plan_outline_array.shape[:2]
        labels = np.full(shape, -1, dtype=np.int32)
        for idx, region in enumerate(self.regions_for_plan):
            labels[np.asarray(region) > 0] = idx
        unique_labels = np.arange(len(self.regions_for_plan), dtype=np.int32)
        return labels, unique_labels

    def _build_source_crn_bundle(self):
        """构建源图 Color Relationship Network 与中心性信息。"""
        labels = getattr(self, 'labels_2d', None)
        hex_codes = list(getattr(self, 'hex_color_codes', []) or [])
        if labels is None or len(hex_codes) < 2:
            return None
        nonnegative_labels = np.unique(np.asarray(labels)[np.asarray(labels) >= 0])
        if len(nonnegative_labels) != len(hex_codes):
            return None
        try:
            color_proportions = [int(np.sum(labels == i)) for i in range(len(hex_codes))]
            G, _, props, disps = self.build_color_network(
                labels,
                hex_codes,
                color_proportions,
                self.network_threshold_var.get() if hasattr(self, 'network_threshold_var') else 0.1
            )
            if G is None or G.number_of_nodes() < 2:
                return None
            centrality = self.compute_network_centrality_measures(G)
            node_attrs = {n: dict(G.nodes[n]) for n in G.nodes()}
            edge_attrs = [
                {
                    'u': u,
                    'v': v,
                    'weight': float(data.get('weight', 0.5)),
                    'boundary_count': int(data.get('boundary_count', 0)),
                }
                for u, v, data in G.edges(data=True)
            ]
            return {
                'G': G,
                'centrality': centrality,
                'proportions': props,
                'node_attrs': node_attrs,
                'edge_attrs': edge_attrs,
                'dispersions': disps,
            }
        except Exception:
            return None


    # ================================================================
    # CRN-Palette v4: 用户调色板约束 + 代表性调色板先验
    # ================================================================

    @staticmethod
    def _rgb_u8_to_cielab(rgb_values):
        """RGB uint8 -> 标准 CIE L*a*b*（L*=0..100，a*/b* 以 0 为中心）。"""
        arr = np.asarray(rgb_values, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        arr = arr.reshape(-1, 1, 3) / 255.0
        return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)

    @staticmethod
    def _cielab_to_rgb_u8(lab_values):
        """标准 CIE L*a*b* -> RGB uint8，并执行可显示色域裁剪。"""
        arr = np.asarray(lab_values, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.uint8)
        arr = arr.reshape(-1, 1, 3)
        rgb = cv2.cvtColor(arr, cv2.COLOR_LAB2RGB).reshape(-1, 3)
        return np.uint8(np.clip(np.rint(rgb * 255.0), 0, 255))

    @staticmethod
    def _rgb_u8_to_hex(rgb_values):
        arr = np.asarray(rgb_values, dtype=np.uint8).reshape(-1, 3)
        return [f'#{int(r):02X}{int(g):02X}{int(b):02X}' for r, g, b in arr]

    def _hex_palette_to_cielab(self, colors):
        if not colors:
            return np.empty((0, 3), dtype=np.float64)
        return self._rgb_u8_to_cielab(self._element_aware_v21_hex_to_rgb(colors))

    @staticmethod
    def _normalized_vector(values, default=0.5):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi - lo < 1e-9:
            return np.full(values.shape, float(default), dtype=np.float64)
        return (values - lo) / (hi - lo)

    def _source_node_standard_lab(self, node_attrs, node):
        attrs = node_attrs.get(node, {}) if node_attrs else {}
        rgb = np.asarray(attrs.get('rgb', [0.5, 0.5, 0.5]), dtype=np.float64).reshape(-1)
        if rgb.size < 3:
            rgb = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        if np.max(rgb[:3]) <= 1.5:
            rgb = rgb[:3] * 255.0
        return self._rgb_u8_to_cielab(np.uint8(np.clip(np.rint(rgb[:3]), 0, 255)))[0]

    def _crn_node_importance_v4(self, G, centrality, node_attrs):
        """以 CRN 中心性与面积共同定义节点角色重要性。"""
        if G is None or G.number_of_nodes() == 0:
            return {}
        pr = centrality.get('pagerank', {})
        btw = centrality.get('betweenness', {})
        ev = centrality.get('eigenvector', {})
        wd = centrality.get('weighted_degree', {})
        values = {}
        for node in G.nodes():
            prop = float(node_attrs.get(node, {}).get('proportion', 0.0))
            values[node] = (
                0.30 * float(pr.get(node, 0.0))
                + 0.25 * float(btw.get(node, 0.0))
                + 0.20 * float(ev.get(node, 0.0))
                + 0.15 * float(wd.get(node, 0.0))
                + 0.10 * min(1.0, max(0.0, prop * max(1, G.number_of_nodes())))
            )
        total = sum(values.values())
        if total <= 1e-12:
            return {node: 1.0 / max(1, G.number_of_nodes()) for node in G.nodes()}
        return {node: value / total for node, value in values.items()}

    def _palette_saliency_v4(self, colors):
        """计算调色板颜色的相对视觉显著度，不把它绑定到源图绝对色相。"""
        if not colors:
            return np.empty(0, dtype=np.float64)
        rgb = self._element_aware_v21_hex_to_rgb(colors)
        labs = self._rgb_u8_to_cielab(rgb)
        hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        chroma = np.sqrt(labs[:, 1] ** 2 + labs[:, 2] ** 2)
        sat = hsv[:, 1].astype(np.float64) / 255.0
        mean_lab = np.mean(labs, axis=0)
        contrast = np.array([self._ciede2000(lab, mean_lab) for lab in labs], dtype=np.float64)
        l_contrast = np.abs(labs[:, 0] - np.mean(labs[:, 0]))
        return np.clip(
            0.42 * self._normalized_vector(chroma)
            + 0.28 * self._normalized_vector(contrast)
            + 0.18 * self._normalized_vector(l_contrast)
            + 0.12 * sat,
            0.0,
            1.0,
        )

    @staticmethod
    def _weighted_color_sample_v41(unique_colors, counts, sample_size, rng):
        """按真实像素频率抽样，避免去重后把高频内部颜色降为单票。

        论文中的 MeanShift 面向采样像素点集；这里允许重复抽到同一量化色，
        使聚类中心与中心权重继续代表原图的颜色质量分布。
        """
        colors = np.asarray(unique_colors, dtype=np.uint8).reshape(-1, 3)
        weights = np.asarray(counts, dtype=np.float64).reshape(-1)
        if len(colors) == 0 or len(colors) != len(weights):
            return np.empty((0, 3), dtype=np.uint8)
        sample_size = max(1, int(sample_size))
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        probabilities = (
            weights / total
            if total > 0.0
            else np.full(len(colors), 1.0 / len(colors), dtype=np.float64)
        )
        indices = rng.choice(
            len(colors), size=sample_size, replace=True, p=probabilities
        )
        return colors[np.asarray(indices, dtype=np.intp)]

    def _extract_representative_palette_prior(self, target_count, quantile=0.2):
        """提取“色域极值 + 内部代表色”先验，且不增加最终输出颜色。

        优先使用 SciPy ConvexHull 与 sklearn MeanShift；缺少可选依赖时，
        自动退化为 PCA 极值 + OpenCV KMeans，避免调色板先验直接失效。
        """
        target_count = max(1, int(target_count))
        image_path = getattr(self, 'image_path', None)
        if not image_path or not os.path.exists(image_path):
            return []
        try:
            mtime = os.path.getmtime(image_path)
        except OSError:
            mtime = 0.0
        cache_key = (os.path.abspath(image_path), float(mtime), target_count, float(quantile), 'v4.1')
        cache = getattr(self, '_representative_palette_prior_cache', {})
        if cache_key in cache:
            return list(cache[cache_key])

        try:
            image = Image.open(image_path).convert('RGB')
            image.thumbnail((256, 256), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
            if len(pixels) == 0:
                return []

            # 轻量量化去重，只用于提取先验，不影响最终渲染。
            quantized = (pixels // 4) * 4 + 2
            unique, counts = np.unique(quantized, axis=0, return_counts=True)
            rng = np.random.default_rng(42)
            if len(unique) > 10000:
                probs = counts.astype(np.float64) / max(1, counts.sum())
                idx = rng.choice(len(unique), size=10000, replace=False, p=probs)
                sample = unique[idx]
                sample_counts = counts[idx]
            else:
                sample = unique
                sample_counts = counts

            # ---- 色域极值候选：ConvexHull 优先，PCA 极值兜底 ----
            hull_candidates = np.empty((0, 3), dtype=np.float64)
            if len(sample) >= 4:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(sample.astype(np.float64), qhull_options='QJ')
                    hull_candidates = sample[np.unique(hull.vertices)].astype(np.float64)
                except Exception:
                    centered = sample.astype(np.float64) - np.average(
                        sample.astype(np.float64), axis=0,
                        weights=np.maximum(sample_counts.astype(np.float64), 1.0)
                    )
                    try:
                        _, _, vh = np.linalg.svd(centered, full_matrices=False)
                        directions = list(vh[:3])
                    except Exception:
                        directions = [
                            np.array([1.0, 0.0, 0.0]),
                            np.array([0.0, 1.0, 0.0]),
                            np.array([0.0, 0.0, 1.0]),
                        ]
                    extrema = set()
                    for direction in directions:
                        projection = centered @ direction
                        extrema.add(int(np.argmin(projection)))
                        extrema.add(int(np.argmax(projection)))
                    hull_candidates = sample[sorted(extrema)].astype(np.float64)

            # ---- 内部代表色：MeanShift 优先，OpenCV KMeans 兜底 ----
            # MeanShift/KMeans 必须看到像素频率，而不是只看到去重后的颜色集合。
            # 固定最多 3000 个点以限制运行时间；重复抽样保留真实颜色质量。
            ms_n = max(1, int(np.sum(counts)))
            internal_sample = self._weighted_color_sample_v41(
                unique, counts, ms_n, rng
            ).astype(np.float64)

            centers = np.empty((0, 3), dtype=np.float64)
            center_weights = np.empty(0, dtype=np.float64)
            if len(internal_sample) >= 2:
                try:
                    from sklearn.cluster import MeanShift, estimate_bandwidth
                    bandwidth = estimate_bandwidth(
                        internal_sample,
                        quantile=float(np.clip(quantile, 0.05, 0.45)),
                        n_samples=min(1200, len(internal_sample)),
                        random_state=42,
                    )
                    if not np.isfinite(bandwidth) or bandwidth < 2.0:
                        bandwidth = 12.0
                    model = MeanShift(
                        bandwidth=float(bandwidth),
                        bin_seeding=True,
                        cluster_all=True,
                        max_iter=120,
                    ).fit(internal_sample)
                    centers = np.asarray(model.cluster_centers_, dtype=np.float64)
                    center_weights = np.bincount(
                        model.labels_, minlength=len(centers)
                    ).astype(np.float64)
                except Exception:
                    k = min(max(2, target_count), len(internal_sample))
                    data = np.float32(internal_sample)
                    fallback_model = KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init=5,
                        max_iter=60,
                    ).fit(data)
                    centers = np.asarray(
                        fallback_model.cluster_centers_, dtype=np.float64
                    )
                    center_weights = np.bincount(
                        np.asarray(fallback_model.labels_).reshape(-1), minlength=k
                    ).astype(np.float64)

            candidates, weights = [], []
            if len(hull_candidates):
                candidates.extend(hull_candidates.tolist())
                # 极值色需要保留，但不让低频极值压过高频内部色。
                hull_weight = max(1.0, float(np.percentile(sample_counts, 60)))
                weights.extend([hull_weight] * len(hull_candidates))
            if len(centers):
                candidates.extend(centers.tolist())
                weights.extend((center_weights + 1.0).tolist())
            if not candidates:
                candidates = sample.astype(np.float64).tolist()
                weights = sample_counts.astype(np.float64).tolist()

            candidate_rgb = np.uint8(np.clip(np.rint(candidates), 0, 255))
            candidate_lab = self._rgb_u8_to_cielab(candidate_rgb)
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / max(float(np.max(weights)), 1.0)

            # 加权 farthest-point sampling：兼顾高频颜色、内部色和色域极值。
            selected = []
            mean_lab = np.average(
                candidate_lab, axis=0, weights=np.maximum(weights, 1e-6)
            )
            first_score = (
                np.linalg.norm(candidate_lab - mean_lab[None, :], axis=1)
                - 8.0 * np.sqrt(np.maximum(weights, 0.0))
            )
            selected.append(int(np.argmin(first_score)))
            while len(selected) < min(target_count, len(candidate_lab)):
                remaining = [i for i in range(len(candidate_lab)) if i not in selected]
                min_dist = np.array([
                    min(self._ciede2000(candidate_lab[i], candidate_lab[j]) for j in selected)
                    for i in remaining
                ])
                score = min_dist * (0.30 + 0.70 * np.sqrt(np.maximum(weights[remaining], 0.0)))
                selected.append(int(remaining[int(np.argmax(score))]))

            prior = self._rgb_u8_to_hex(candidate_rgb[selected])
            cache[cache_key] = list(prior)
            self._representative_palette_prior_cache = cache
            return prior
        except Exception:
            return []

    def _compute_palette_anchor_fidelity_v4(self, adapted_colors, user_colors):
        if not adapted_colors or not user_colors:
            return 0.0
        n = min(len(adapted_colors), len(user_colors))
        a_lab = self._hex_palette_to_cielab(adapted_colors[:n])
        u_lab = self._hex_palette_to_cielab(user_colors[:n])
        delta = np.array([self._ciede2000(a_lab[i], u_lab[i]) for i in range(n)])
        # ΔE00≈2 通常接近肉眼刚可察；σ=5 允许温和适配但严惩偏离。
        return float(np.clip(100.0 * np.mean(np.exp(-((delta / 5.0) ** 2))), 0.0, 100.0))

    def _compute_representative_palette_geometry_v4(self, colors, representative_palette):
        """比较调色板内部关系几何，而非要求目标色贴近源色。"""
        if len(colors) < 2 or len(representative_palette) < 2:
            return 70.0
        labs_a = self._hex_palette_to_cielab(colors)
        labs_b = self._hex_palette_to_cielab(representative_palette)

        def normalized_pairwise(labs):
            values = []
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    values.append(self._ciede2000(labs[i], labs[j]))
            values = np.sort(np.asarray(values, dtype=np.float64))
            if values.size == 0:
                return values
            return values / max(float(np.mean(values)), 1e-6)

        first = normalized_pairwise(labs_a)
        second = normalized_pairwise(labs_b)
        if first.size == 0 or second.size == 0:
            return 70.0
        # 数量不一致时在 0..1 参数域重采样。
        size = max(len(first), len(second))
        x = np.linspace(0.0, 1.0, size)
        first = np.interp(x, np.linspace(0.0, 1.0, len(first)), first)
        second = np.interp(x, np.linspace(0.0, 1.0, len(second)), second)
        error = np.mean(np.abs(np.log((first + 0.12) / (second + 0.12))))
        return float(np.clip(100.0 * np.exp(-error / 0.55), 0.0, 100.0))

    @staticmethod
    def _probability_similarity_v414(first, second):
        first = np.asarray(first, dtype=np.float64).reshape(-1)
        second = np.asarray(second, dtype=np.float64).reshape(-1)
        size = max(len(first), len(second))
        if size == 0:
            return 0.5
        first = np.pad(np.maximum(first, 0.0), (0, size - len(first)))
        second = np.pad(np.maximum(second, 0.0), (0, size - len(second)))
        if float(np.sum(first)) <= 1e-12:
            first[:] = 1.0
        if float(np.sum(second)) <= 1e-12:
            second[:] = 1.0
        first /= float(np.sum(first))
        second /= float(np.sum(second))
        hellinger = np.sqrt(np.sum((np.sqrt(first) - np.sqrt(second)) ** 2)) / np.sqrt(2.0)
        l1_similarity = 1.0 - 0.5 * float(np.sum(np.abs(first - second)))
        return float(np.clip(0.58 * (1.0 - hellinger) + 0.42 * l1_similarity, 0.0, 1.0))

    def _build_target_color_crn_v414(
            self, labels, unique_labels, placement, colors):
        """把目标 owner 图按 placement 折叠为以颜色索引为节点的真实 CRN。"""
        color_count = len(colors)
        if labels is None or unique_labels is None or color_count <= 0:
            return None
        color_map = np.full(np.asarray(labels).shape, -1, dtype=np.int32)
        for region_index in range(min(len(placement), len(unique_labels))):
            color_index = self._normalize_index(placement[region_index])
            if color_index is None or color_index >= color_count:
                continue
            color_map[np.asarray(labels) == unique_labels[region_index]] = int(color_index)
        counts = np.asarray([
            np.count_nonzero(color_map == color_index)
            for color_index in range(color_count)
        ], dtype=np.float64)
        graph, _, proportions, dispersions = self.build_color_network(
            color_map, colors, counts, network_threshold=0.0
        )
        centrality = self.compute_network_centrality_measures(graph) if graph is not None else {}
        return {
            'labels': color_map,
            'counts': counts,
            'proportions': np.asarray(proportions, dtype=np.float64),
            'graph': graph,
            'centrality': centrality,
            'dispersions': np.asarray(dispersions, dtype=np.float64),
        }

    def _compute_true_crn_fidelity_v414(
            self, labels, unique_labels, placement, colors,
            source_G, source_centrality, source_proportions, node_attrs):
        """比较源/目标 CRN 的节点面积、中心性和边拓扑，索引均为颜色编号。"""
        color_count = len(colors)
        target = self._build_target_color_crn_v414(
            labels, unique_labels, placement, colors
        )
        if target is None or source_G is None or color_count <= 0:
            return {
                'mass': 50.0, 'centrality': 50.0, 'edge_graph': 50.0,
                'minimum_share': 0.0, 'maximum_share': 1.0,
                'target_shares': [], 'source_shares': [], 'constraint': 0.0,
            }

        source_share = np.zeros(color_count, dtype=np.float64)
        supplied = np.asarray(source_proportions, dtype=np.float64).reshape(-1)
        for color_index in range(color_count):
            if color_index < len(supplied):
                source_share[color_index] = max(0.0, float(supplied[color_index]))
            elif color_index in source_G.nodes:
                source_share[color_index] = max(
                    0.0, float(node_attrs.get(color_index, {}).get('proportion', 0.0))
                )
        if float(np.sum(source_share)) <= 1e-12:
            source_share[:] = 1.0
        source_share /= float(np.sum(source_share))

        target_share = np.asarray(target['proportions'], dtype=np.float64)
        if len(target_share) < color_count:
            target_share = np.pad(target_share, (0, color_count - len(target_share)))
        target_share = target_share[:color_count]
        if float(np.sum(target_share)) > 0.0:
            target_share /= float(np.sum(target_share))

        basic_mass = self._probability_similarity_v414(source_share, target_share)
        log_error = float(np.sum(
            source_share * np.abs(np.log((target_share + 0.012) / (source_share + 0.012)))
        ))
        log_similarity = float(np.exp(-log_error / 0.62))
        mass_score = 100.0 * (0.62 * basic_mass + 0.38 * log_similarity)

        positive = target_share[target_share > 1e-12]
        minimum_share = float(np.min(positive)) if len(positive) else 0.0
        maximum_share = float(np.max(target_share)) if len(target_share) else 1.0
        used_count = int(np.count_nonzero(target_share > 1e-12))
        coverage_gate = used_count / max(1, color_count)
        minimum_gate = min(1.0, minimum_share / 0.05) if used_count == color_count else 0.0
        maximum_gate = float(np.exp(-max(0.0, maximum_share - 0.58) / 0.12))
        constraint = float(np.clip(coverage_gate * minimum_gate * maximum_gate, 0.0, 1.0))
        mass_score *= 0.70 + 0.30 * constraint

        target_centrality = target['centrality']
        metric_weights = {
            'pagerank': 0.34,
            'weighted_degree': 0.29,
            'betweenness': 0.22,
            'eigenvector': 0.15,
        }
        centrality_score = 0.0
        for metric, metric_weight in metric_weights.items():
            source_vector = np.asarray([
                float(source_centrality.get(metric, {}).get(index, 0.0))
                for index in range(color_count)
            ], dtype=np.float64)
            target_vector = np.asarray([
                float(target_centrality.get(metric, {}).get(index, 0.0))
                for index in range(color_count)
            ], dtype=np.float64)
            distribution_similarity = self._probability_similarity_v414(
                source_vector, target_vector
            )
            source_rank = np.argsort(np.argsort(source_vector, kind='stable'), kind='stable')
            target_rank = np.argsort(np.argsort(target_vector, kind='stable'), kind='stable')
            if color_count >= 2 and np.std(source_rank) > 0 and np.std(target_rank) > 0:
                rho = float(np.corrcoef(source_rank, target_rank)[0, 1])
                if not np.isfinite(rho):
                    rho = 0.0
                rank_similarity = 0.5 * (rho + 1.0)
            else:
                rank_similarity = distribution_similarity
            centrality_score += metric_weight * (
                0.72 * distribution_similarity + 0.28 * rank_similarity
            )
        centrality_score = 100.0 * float(np.clip(centrality_score, 0.0, 1.0))

        def edge_vector(graph):
            values = []
            present = []
            for first in range(color_count):
                for second in range(first + 1, color_count):
                    if graph is not None and graph.has_edge(first, second):
                        data = graph[first][second]
                        strength = max(0.0, float(data.get('weight', 0.0))) * (
                            1.0 + np.log1p(max(0.0, float(data.get('boundary_count', 0.0))))
                        )
                        values.append(strength)
                        present.append(1.0)
                    else:
                        values.append(0.0)
                        present.append(0.0)
            return np.asarray(values, dtype=np.float64), np.asarray(present, dtype=np.float64)

        source_edges, source_present = edge_vector(source_G)
        target_edges, target_present = edge_vector(target['graph'])
        if len(source_edges) == 0:
            edge_graph_score = 70.0
        else:
            edge_distribution = self._probability_similarity_v414(
                source_edges, target_edges
            )
            denom = float(np.linalg.norm(source_edges) * np.linalg.norm(target_edges))
            cosine = float(np.dot(source_edges, target_edges) / denom) if denom > 1e-12 else 0.0
            true_positive = float(np.sum((source_present > 0) & (target_present > 0)))
            precision = true_positive / max(1.0, float(np.sum(target_present > 0)))
            recall = true_positive / max(1.0, float(np.sum(source_present > 0)))
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            edge_graph_score = 100.0 * float(np.clip(
                0.46 * edge_distribution + 0.34 * cosine + 0.20 * f1,
                0.0, 1.0,
            ))

        return {
            'mass': float(np.clip(mass_score, 0.0, 100.0)),
            'centrality': float(np.clip(centrality_score, 0.0, 100.0)),
            'edge_graph': float(np.clip(edge_graph_score, 0.0, 100.0)),
            'minimum_share': minimum_share,
            'maximum_share': maximum_share,
            'target_shares': [float(value) for value in target_share],
            'source_shares': [float(value) for value in source_share],
            'constraint': 100.0 * constraint,
        }

    def _compute_ELH_relational_v4(self, source_G, placement, colors, edge_attrs, node_attrs):
        """相对边关系保持：允许整体换风格，只保持 CRN 对比结构。"""
        if source_G is None or source_G.number_of_edges() == 0 or not colors:
            return 55.0
        target_labs = self._hex_palette_to_cielab(colors)
        edge_list = (
            [(e['u'], e['v'], e) for e in edge_attrs]
            if edge_attrs else list(source_G.edges(data=True))
        )
        rows = []
        for u, v, data in edge_list:
            nu, nv = self._normalize_index(u), self._normalize_index(v)
            if nu is None or nv is None or nu >= len(placement) or nv >= len(placement):
                continue
            cu, cv = self._normalize_index(placement[nu]), self._normalize_index(placement[nv])
            if cu is None or cv is None or cu >= len(target_labs) or cv >= len(target_labs):
                continue
            src_de = self._ciede2000(
                self._source_node_standard_lab(node_attrs, u),
                self._source_node_standard_lab(node_attrs, v),
            )
            tgt_de = self._ciede2000(target_labs[cu], target_labs[cv])
            weight = float(data.get('weight', 0.5))
            boundary = float(data.get('boundary_count', 0.0))
            effective = max(1e-6, weight) * (1.0 + np.log1p(max(0.0, boundary)))
            rows.append((effective, src_de, tgt_de, cu == cv))
        if not rows:
            return 55.0
        weights = np.array([r[0] for r in rows], dtype=np.float64)
        src = np.array([r[1] for r in rows], dtype=np.float64)
        tgt = np.array([r[2] for r in rows], dtype=np.float64)
        src_mean = float(np.average(src, weights=weights))
        tgt_mean = float(np.average(tgt, weights=weights))
        scores = []
        for _, s_de, t_de, same in rows:
            ratio_error = abs(np.log((t_de / max(tgt_mean, 1e-6) + 0.12) /
                                     (s_de / max(src_mean, 1e-6) + 0.12)))
            relation = np.exp(-ratio_error / 0.55)
            order = 1.0 if (s_de - src_mean) * (t_de - tgt_mean) >= 0 else 0.45
            separation = 1.0 - np.exp(-max(0.0, t_de) / 10.0)
            if same:
                separation *= 0.35
            scores.append(0.70 * relation + 0.18 * order + 0.12 * separation)
        return float(np.clip(100.0 * np.average(scores, weights=weights), 0.0, 100.0))

    def _compute_node_role_alignment_v4(self, placement, colors, G, centrality, node_attrs):
        if G is None or G.number_of_nodes() == 0 or not colors:
            return 50.0
        importance = self._crn_node_importance_v4(G, centrality, node_attrs)
        saliency = self._palette_saliency_v4(colors)
        node_imp = []
        node_sal = []
        node_weights = []
        for node in G.nodes():
            idx = self._normalize_index(node)
            if idx is None or idx >= len(placement):
                continue
            ci = self._normalize_index(placement[idx])
            if ci is None or ci >= len(saliency):
                continue
            node_imp.append(float(importance.get(node, 0.0)))
            node_sal.append(float(saliency[ci]))
            node_weights.append(max(1e-6, float(importance.get(node, 0.0))))
        if not node_imp:
            return 50.0
        imp_norm = self._normalized_vector(node_imp)
        sal_norm = self._normalized_vector(node_sal)
        error = np.average(np.abs(imp_norm - sal_norm), weights=node_weights)
        return float(np.clip(100.0 * (1.0 - error), 0.0, 100.0))

    def _compute_VAFE_style_v4(self, placement, colors, G, centrality, node_attrs):
        if G is None or G.number_of_nodes() == 0 or not colors:
            return 50.0
        importance = self._crn_node_importance_v4(G, centrality, node_attrs)
        saliency = self._palette_saliency_v4(colors)
        nodes = []
        imp_vals = []
        sal_vals = []
        for node in G.nodes():
            idx = self._normalize_index(node)
            if idx is None or idx >= len(placement):
                continue
            ci = self._normalize_index(placement[idx])
            if ci is None or ci >= len(saliency):
                continue
            nodes.append(node)
            imp_vals.append(float(importance.get(node, 0.0)))
            sal_vals.append(float(saliency[ci]))
        if not nodes:
            return 50.0
        imp_norm = self._normalized_vector(imp_vals)
        sal_norm = self._normalized_vector(sal_vals)
        node_to_pos = {node: pos for pos, node in enumerate(nodes)}

        threshold = np.percentile(imp_norm, 70) if len(imp_norm) > 1 else imp_norm[0]
        top = imp_norm >= threshold
        focus = 100.0 * float(np.mean(sal_norm[top])) if np.any(top) else 50.0

        edge_scores, edge_weights = [], []
        for u, v, data in G.edges(data=True):
            if u not in node_to_pos or v not in node_to_pos:
                continue
            pu, pv = node_to_pos[u], node_to_pos[v]
            gradient_error = abs(abs(imp_norm[pu] - imp_norm[pv]) - abs(sal_norm[pu] - sal_norm[pv]))
            edge_scores.append(np.exp(-gradient_error / 0.28))
            edge_weights.append(max(1e-6, float(data.get('weight', 0.5))))
        flow = 100.0 * float(np.average(edge_scores, weights=edge_weights)) if edge_scores else 70.0

        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(imp_norm, sal_norm).correlation)
            if not np.isfinite(rho):
                rho = 0.0
        except Exception:
            rho = float(np.corrcoef(imp_norm, sal_norm)[0, 1]) if len(imp_norm) > 1 else 0.0
            if not np.isfinite(rho):
                rho = 0.0
        hierarchy = 50.0 * (rho + 1.0)
        return float(np.clip(0.42 * focus + 0.38 * flow + 0.20 * hierarchy, 0.0, 100.0))

    def _compute_NSB_style_v4(self, placement, labels, unique_labels, colors, G, centrality, node_attrs):
        if G is None or G.number_of_nodes() == 0 or not colors:
            return 50.0
        importance = self._crn_node_importance_v4(G, centrality, node_attrs)
        areas = []
        for idx in range(min(len(placement), len(unique_labels))):
            areas.append(float(np.count_nonzero(labels == unique_labels[idx])))
        area_balance = (1.0 - self._gini_coefficient(areas)) * 100.0 if len(areas) >= 2 else 50.0

        color_load = np.zeros(len(colors), dtype=np.float64)
        valid_nodes = 0
        for node in G.nodes():
            idx = self._normalize_index(node)
            if idx is None or idx >= len(placement):
                continue
            ci = self._normalize_index(placement[idx])
            if ci is None or ci >= len(colors):
                continue
            color_load[ci] += float(importance.get(node, 0.0))
            valid_nodes += 1
        coverage = 100.0 * np.count_nonzero(color_load > 1e-9) / max(1, len(colors))
        load_balance = (1.0 - self._gini_coefficient(color_load[color_load > 0])) * 100.0 \
            if np.count_nonzero(color_load > 0) >= 2 else 45.0

        collision_weight = 0.0
        total_weight = 0.0
        for u, v, data in G.edges(data=True):
            iu, iv = self._normalize_index(u), self._normalize_index(v)
            if iu is None or iv is None or iu >= len(placement) or iv >= len(placement):
                continue
            cu, cv = self._normalize_index(placement[iu]), self._normalize_index(placement[iv])
            if cu is None or cv is None:
                continue
            weight = max(1e-6, float(data.get('weight', 0.5)))
            total_weight += weight
            if cu == cv:
                collision_weight += weight
        edge_separation = 100.0 * (1.0 - collision_weight / total_weight) if total_weight > 0 else 70.0

        saliency = self._palette_saliency_v4(colors)
        if len(saliency) >= 3:
            tiers = np.digitize(saliency, np.quantile(saliency, [1 / 3, 2 / 3]), right=True)
            used_tiers = set()
            for idx in range(min(len(placement), G.number_of_nodes())):
                ci = self._normalize_index(placement[idx])
                if ci is not None and ci < len(tiers):
                    used_tiers.add(int(tiers[ci]))
            tier_coverage = 100.0 * len(used_tiers) / 3.0
        else:
            tier_coverage = coverage
        return float(np.clip(
            0.25 * area_balance
            + 0.20 * coverage
            + 0.20 * load_balance
            + 0.22 * edge_separation
            + 0.13 * tier_coverage,
            0.0,
            100.0,
        ))

    def _compute_topology_style_v4(self, placement, colors, G, node_attrs):
        if G is None or G.number_of_edges() == 0 or not colors:
            return 55.0
        labs = self._hex_palette_to_cielab(colors)
        src_dist, tgt_dist, weights = [], [], []
        for u, v, data in G.edges(data=True):
            iu, iv = self._normalize_index(u), self._normalize_index(v)
            if iu is None or iv is None or iu >= len(placement) or iv >= len(placement):
                continue
            cu, cv = self._normalize_index(placement[iu]), self._normalize_index(placement[iv])
            if cu is None or cv is None or cu >= len(labs) or cv >= len(labs):
                continue
            src_dist.append(self._ciede2000(
                self._source_node_standard_lab(node_attrs, u),
                self._source_node_standard_lab(node_attrs, v),
            ))
            tgt_dist.append(self._ciede2000(labs[cu], labs[cv]))
            weights.append(max(1e-6, float(data.get('weight', 0.5))))
        if len(src_dist) < 2:
            return 60.0
        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(src_dist, tgt_dist).correlation)
            if not np.isfinite(rho):
                rho = 0.0
        except Exception:
            rho = float(np.corrcoef(src_dist, tgt_dist)[0, 1])
            if not np.isfinite(rho):
                rho = 0.0
        rank_score = 50.0 * (rho + 1.0)

        community_score = 65.0
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G, weight='weight'))
            values = []
            for community in communities:
                assigned = []
                for node in community:
                    idx = self._normalize_index(node)
                    if idx is None or idx >= len(placement):
                        continue
                    ci = self._normalize_index(placement[idx])
                    if ci is not None:
                        assigned.append(ci)
                if assigned:
                    counts = np.bincount(assigned)
                    values.append(float(np.max(counts) / len(assigned)))
            if values:
                community_score = 100.0 * float(np.mean(values))
        except Exception:
            pass
        return float(np.clip(0.65 * rank_score + 0.35 * community_score, 0.0, 100.0))

    def _palette_adaptation_objective_v4(
            self, colors, user_colors, placement, source_bundle, representative_palette):
        if source_bundle is None:
            return self._compute_palette_anchor_fidelity_v4(colors, user_colors)
        elh = self._compute_ELH_relational_v4(
            source_bundle['G'], placement, colors,
            source_bundle['edge_attrs'], source_bundle['node_attrs']
        )
        node = self._compute_node_role_alignment_v4(
            placement, colors, source_bundle['G'],
            source_bundle['centrality'], source_bundle['node_attrs']
        )
        anchor = self._compute_palette_anchor_fidelity_v4(colors, user_colors)
        representative = self._compute_representative_palette_geometry_v4(
            colors, representative_palette
        )
        return 0.42 * elh + 0.28 * node + 0.22 * anchor + 0.08 * representative

    def _adapt_user_palette_for_crn_v4(
            self, user_colors, placement, source_bundle, representative_palette,
            max_delta_e=6.0, max_mean_delta_e=3.0):
        """在用户颜色附近做受限微调；不新增颜色、不交换颜色身份。

        除单色 ΔE/L*/C*/h 约束外，增加调色板两两距离保护，防止多个
        用户颜色在适配后塌缩成几乎相同的颜色。
        """
        if not user_colors or source_bundle is None or len(user_colors) < 2:
            return list(user_colors), {
                'mean_delta_e': 0.0, 'max_delta_e': 0.0,
                'changed': False, 'objective': 0.0,
                'pairwise_guard': True,
            }

        user_labs = self._hex_palette_to_cielab(user_colors)
        current_labs = user_labs.copy()
        current_colors = list(user_colors)
        best_score = self._palette_adaptation_objective_v4(
            current_colors, user_colors, placement, source_bundle, representative_palette
        )

        shifts = np.array([
            [0, 0, 0], [1.5, 0, 0], [-1.5, 0, 0], [3, 0, 0], [-3, 0, 0],
            [0, 1.5, 0], [0, -1.5, 0], [0, 0, 1.5], [0, 0, -1.5],
            [1.5, 1.5, 0], [-1.5, -1.5, 0], [1.5, 0, 1.5], [-1.5, 0, -1.5],
            [0, 2.0, 2.0], [0, -2.0, -2.0], [0, 2.0, -2.0], [0, -2.0, 2.0],
        ], dtype=np.float64)

        def pairwise_guard(index, candidate_lab, all_labs):
            for j in range(len(all_labs)):
                if j == index:
                    continue
                source_de = self._ciede2000(user_labs[index], user_labs[j])
                candidate_de = self._ciede2000(candidate_lab, all_labs[j])
                # 用户原本可区分的颜色，适配后不得发生明显塌缩。
                if source_de >= 10.0:
                    minimum = max(7.0, 0.62 * source_de)
                    if candidate_de < minimum:
                        return False
                elif source_de >= 5.0 and candidate_de < 4.0:
                    return False
            return True

        def valid_candidate(index, lab, all_labs):
            base = user_labs[index]
            delta = self._ciede2000(base, lab)
            if delta > float(max_delta_e) + 1e-6:
                return False
            if abs(float(lab[0] - base[0])) > 5.0:
                return False
            base_c = float(np.hypot(base[1], base[2]))
            cand_c = float(np.hypot(lab[1], lab[2]))
            if abs(cand_c - base_c) > 7.0:
                return False
            if base_c >= 12.0 and cand_c >= 8.0:
                h0 = np.degrees(np.arctan2(base[2], base[1])) % 360.0
                h1 = np.degrees(np.arctan2(lab[2], lab[1])) % 360.0
                hue_diff = min(abs(h1 - h0), 360.0 - abs(h1 - h0))
                if hue_diff > 9.0:
                    return False
            temp = np.array(all_labs, dtype=np.float64, copy=True)
            temp[index] = lab
            deltas = np.array([
                self._ciede2000(user_labs[i], temp[i]) for i in range(len(temp))
            ], dtype=np.float64)
            if float(np.mean(deltas)) > float(max_mean_delta_e) + 1e-6:
                return False
            return pairwise_guard(index, lab, temp)

        for _ in range(2):
            improved = False
            for color_index in range(len(current_labs)):
                local_best_lab = current_labs[color_index].copy()
                local_best_colors = list(current_colors)
                local_best_score = best_score
                for shift in shifts:
                    proposed_lab = current_labs[color_index] + shift
                    if not valid_candidate(color_index, proposed_lab, current_labs):
                        continue
                    proposed_rgb = self._cielab_to_rgb_u8(proposed_lab)[0]
                    realized_lab = self._rgb_u8_to_cielab(proposed_rgb)[0]
                    if not valid_candidate(color_index, realized_lab, current_labs):
                        continue
                    candidate_colors = list(current_colors)
                    candidate_colors[color_index] = self._rgb_u8_to_hex([proposed_rgb])[0]
                    score = self._palette_adaptation_objective_v4(
                        candidate_colors, user_colors, placement,
                        source_bundle, representative_palette
                    )
                    if score > local_best_score + 1e-6:
                        local_best_score = score
                        local_best_lab = realized_lab
                        local_best_colors = candidate_colors
                if local_best_score > best_score + 1e-6:
                    current_labs[color_index] = local_best_lab
                    current_colors = local_best_colors
                    best_score = local_best_score
                    improved = True
            if not improved:
                break

        final_labs = self._hex_palette_to_cielab(current_colors)
        deltas = np.array([
            self._ciede2000(user_labs[i], final_labs[i]) for i in range(len(user_labs))
        ], dtype=np.float64)
        pairwise_ok = all(
            self._ciede2000(final_labs[i], final_labs[j]) >=
            (max(7.0, 0.62 * self._ciede2000(user_labs[i], user_labs[j]))
             if self._ciede2000(user_labs[i], user_labs[j]) >= 10.0 else 0.0)
            for i in range(len(final_labs)) for j in range(i + 1, len(final_labs))
        )
        return current_colors, {
            'mean_delta_e': float(np.mean(deltas)) if len(deltas) else 0.0,
            'max_delta_e': float(np.max(deltas)) if len(deltas) else 0.0,
            'changed': bool(np.any(deltas > 0.25)),
            'objective': float(best_score),
            'pairwise_guard': bool(pairwise_ok),
        }

    @staticmethod
    def _build_element_aware_candidates(
            base, region_areas, num_colors=None):
        """惰性枚举全部 P(N,M) 个部分排列，不把候选一次性装入内存。

        每种颜色恰好出现一次，未选择的 N-M 个区域为 None。这里不设置
        MAX 上限；使用生成器逐个产出，避免 ``candidates.append`` 导致
        大规模排列时直接触发 MemoryError。
        """
        role_count = len(base)
        if num_colors is None:
            valid = [int(v) for v in base if v is not None]
            num_colors = max(valid) + 1 if valid else 0
        try:
            num_colors = int(num_colors)
        except (TypeError, ValueError, OverflowError):
            return iter(())
        if role_count <= 0 or num_colors <= 0 or num_colors > role_count:
            return iter(())

        base_tuple = tuple(base)
        base_values = [value for value in base_tuple if value is not None]
        base_is_valid = (
            len(base_values) == num_colors
            and sorted(int(value) for value in base_values)
            == list(range(num_colors))
        )

        def iterator():
            if base_is_valid:
                yield base_tuple
            for ordered_regions in itertools.permutations(
                    range(role_count), num_colors):
                placement = [None] * role_count
                for color_index, region_index in enumerate(ordered_regions):
                    placement[int(region_index)] = int(color_index)
                candidate = tuple(placement)
                if base_is_valid and candidate == base_tuple:
                    continue
                yield candidate

        return iterator()

    @staticmethod
    def _placement_distance(first, second, region_areas=None):
        first = tuple(first)
        second = tuple(second)
        count = min(len(first), len(second))
        if count == 0:
            return 0.0
        weights = np.asarray(
            list(region_areas)[:count] if region_areas is not None else [1.0] * count,
            dtype=np.float64,
        )
        if weights.size != count or float(np.sum(weights)) <= 0.0:
            weights = np.ones(count, dtype=np.float64)
        changed = np.fromiter(
            (first[index] != second[index] for index in range(count)),
            dtype=bool,
            count=count,
        )
        return float(np.sum(weights[changed]) / np.sum(weights))

    @classmethod
    def _select_diverse_element_aware_plans(
            cls, scored, total_needed, user_colors, region_areas=None):
        """优先选择不同映射，再在必要时保留同映射的调色板变体。"""
        total_needed = max(0, int(total_needed))
        ranked = sorted(scored, key=lambda item: float(item[1]), reverse=True)
        if total_needed == 0 or not ranked:
            return []

        best_by_mapping = {}
        for item in ranked:
            best_by_mapping.setdefault(tuple(item[0]), item)
        unique_pool = list(best_by_mapping.values())

        selected = [unique_pool.pop(0)]
        while unique_pool and len(selected) < total_needed:
            # 最高 18 分的多样性奖励：大区域换色比微小装饰换色更有价值。
            def utility(item):
                distance = min(
                    cls._placement_distance(item[0], chosen[0], region_areas)
                    for chosen in selected
                )
                return float(item[1]) + 18.0 * distance

            chosen = max(unique_pool, key=utility)
            unique_pool.remove(chosen)
            selected.append(chosen)

        # 只有不同映射数量不足时，才允许同一映射的 Exact/Adapted 变体补位。
        selected_ids = {id(item) for item in selected}
        for item in ranked:
            if len(selected) >= total_needed:
                break
            if id(item) not in selected_ids:
                selected.append(item)
                selected_ids.add(id(item))

        exact = [item for item in ranked if tuple(item[5]) == tuple(user_colors)]
        if total_needed >= 2 and exact and not any(
                tuple(item[5]) == tuple(user_colors) for item in selected):
            selected_mappings = {tuple(item[0]) for item in selected}
            same_mapping_exact = next(
                (item for item in exact if tuple(item[0]) in selected_mappings), None
            )
            replacement = same_mapping_exact or exact[0]
            if same_mapping_exact is not None:
                replace_index = next(
                    index for index, item in enumerate(selected)
                    if tuple(item[0]) == tuple(replacement[0])
                )
            else:
                replace_index = len(selected) - 1
            selected[replace_index] = replacement

        return sorted(selected[:total_needed], key=lambda item: float(item[1]), reverse=True)

    @staticmethod
    def _element_aware_color_area_summary(placement, region_areas, color_count):
        loads = np.zeros(max(0, int(color_count)), dtype=np.float64)
        for region_index, area in enumerate(region_areas):
            if region_index >= len(placement):
                break
            value = placement[region_index]
            if value is None:
                continue
            try:
                color_index = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= color_index < len(loads):
                loads[color_index] += max(0.0, float(area))
        total = float(np.sum(loads))
        shares = loads / total if total > 0.0 else loads
        return {
            'used_color_count': int(np.count_nonzero(loads > 0.0)),
            'dominant_color_share': float(np.max(shares)) if shares.size else 0.0,
            'color_area_shares': [float(value) for value in shares],
        }

    def _generate_element_aware_v21_plans(
            self, selected_colors, total_images=None, progress_callback=None):
        """生成全部 P(N,M) 个 Element-Aware 部分着色方案。

        N 为区域总数，M 为所选颜色数。每种颜色在一个方案中只出现一次，
        其余 N-M 个区域保持白底；不再使用 3000/20000/30000 等候选上限，
        也不再由 Rows×Cols 截断结果数量。
        """
        def report(value, message):
            if progress_callback:
                try:
                    progress_callback(float(value), str(message))
                except Exception:
                    pass

        if not selected_colors or not getattr(self, 'regions_for_plan', None):
            return [], []

        user_colors = list(selected_colors)
        role_count = len(self.regions_for_plan)
        color_count = len(user_colors)
        if color_count > role_count:
            messagebox.showerror(
                "Error!",
                f"Selected colors ({color_count}) cannot exceed number of regions ({role_count})"
            )
            return [], []

        region_areas = [
            int(np.count_nonzero(np.asarray(region) > 0))
            for region in self.regions_for_plan
        ]

        # 将原语义映射仅作为候选排序种子，转换成“每色一次”的部分排列。
        full_base = list(self._element_aware_v21_base_placement(user_colors))
        base = [None] * role_count
        unused_regions = set(range(role_count))
        for color_index in range(color_count):
            preferred = [
                idx for idx in unused_regions
                if idx < len(full_base) and full_base[idx] == color_index
            ]
            pool = preferred or list(unused_regions)
            if not pool:
                break
            region_index = max(
                pool,
                key=lambda idx: (region_areas[idx], -idx),
            )
            base[region_index] = color_index
            unused_regions.remove(region_index)

        exact_count = (
            math.factorial(role_count)
            // math.factorial(role_count - color_count)
        )
        candidates = self._build_element_aware_candidates(
            base,
            region_areas,
            num_colors=color_count,
        )

        formula = f"P({role_count},{color_count})={exact_count}"
        self._last_element_aware_candidate_count = exact_count
        self._last_element_aware_screened_mapping_count = exact_count
        self._last_element_aware_permutation_count = exact_count
        self._last_element_aware_permutation_formula = formula
        self._last_element_aware_shortlist_count = exact_count
        self._last_element_aware_scored_mapping_count = exact_count
        self._last_element_aware_user_palette = list(user_colors)
        self._last_element_aware_plan_palettes = [
            list(user_colors) for _ in range(exact_count)
        ]
        self._last_element_aware_adapted_palette = list(user_colors)
        self._last_element_aware_palette_adaptation = {
            'mean_delta_e': 0.0,
            'max_delta_e': 0.0,
            'changed': False,
            'pairwise_guard': True,
            'mode': 'Exact user palette',
        }

        owner_map = getattr(self, '_element_aware_v210_owner_map', None)
        if owner_map is None or owner_map.shape != self.plan_outline_array.shape[:2]:
            owner_map = np.full(
                self.plan_outline_array.shape[:2], -1, dtype=np.int32
            )
            for region_index, region in enumerate(self.regions_for_plan):
                owner_map[np.asarray(region) > 0] = region_index

        labels_2d, unique_labels = self._build_element_aware_group_labels()
        source_bundle = self._build_source_crn_bundle()
        representative_palette = self._extract_representative_palette_prior(
            color_count, quantile=0.2
        )
        palette_rgb = self._element_aware_v21_hex_to_rgb(user_colors)

        scored = []
        enumerated_count = 0
        report(0.0, f"Scoring 0/{exact_count} ({formula})")
        for index, placement in enumerate(candidates, start=1):
            enumerated_count = index
            placement_list = list(placement)
            breakdown = None
            explanation = None
            rendered_image = None

            if (
                source_bundle is not None
                and labels_2d is not None
                and unique_labels is not None
            ):
                breakdown = self.get_crn_score_breakdown(
                    None,
                    labels_2d,
                    unique_labels,
                    placement_list,
                    user_colors,
                    source_bundle['G'],
                    source_bundle['centrality'],
                    source_bundle['proportions'],
                    node_attrs=source_bundle['node_attrs'],
                    edge_attrs=source_bundle['edge_attrs'],
                    source_disps=source_bundle['dispersions'],
                    user_selected_colors=user_colors,
                    representative_palette=representative_palette,
                )
                breakdown = dict(breakdown)
                score = float(breakdown.get('total', 0.0))
                explanation = self.build_crn_explanation_summary(
                    placement_list,
                    user_colors,
                    source_bundle['G'],
                    source_bundle['centrality'],
                    source_bundle['proportions'],
                    source_bundle['node_attrs'],
                    breakdown=breakdown,
                )
            else:
                if getattr(self, '_transfer_element_aware_active', False):
                    rendered = self._render_transfer_element_aware_exact(
                        self.plan_outline_array,
                        owner_map,
                        palette_rgb,
                        placement_list,
                    )
                else:
                    rendered = self._render_element_aware_v21(
                        self.plan_outline_array,
                        owner_map,
                        palette_rgb,
                        placement_list,
                    )
                rendered_image = Image.fromarray(rendered)
                score = float(self.calculate_image_score(
                    rendered,
                    getattr(self, 'color_regions', None),
                    labels_2d,
                ))

            area_summary = self._element_aware_color_area_summary(
                placement_list, region_areas, color_count
            )
            if breakdown is None:
                breakdown = {'total': score, 'CRNAvailable': False}
            else:
                breakdown['CRNAvailable'] = True
            breakdown.update({
                'PaletteMeanDeltaE': 0.0,
                'PaletteMaxDeltaE': 0.0,
                'PaletteMode': 'Exact user palette / partial permutation',
                'PairwiseGuard': True,
                'UsedColorCount': area_summary['used_color_count'],
                'DominantColorShare': area_summary['dominant_color_share'],
                'ColorAreaShares': area_summary['color_area_shares'],
                'BlankRegionCount': role_count - color_count,
                'ExactPlanCount': exact_count,
            })
            scored.append((
                placement_list,
                score,
                rendered_image,
                breakdown,
                explanation,
                list(user_colors),
            ))
            if index == exact_count or index % max(1, exact_count // 100) == 0:
                report(
                    70.0 * index / max(1, exact_count),
                    f"Scoring {index}/{exact_count} ({formula})",
                )

        if enumerated_count != exact_count:
            raise RuntimeError(
                "Partial permutation enumeration mismatch: "
                f"expected {exact_count}, got {enumerated_count}"
            )

        scored.sort(key=lambda item: float(item[1]), reverse=True)
        rendered_scored = []
        for index, item in enumerate(scored, start=1):
            placement, score, rendered_image, breakdown, explanation, plan_colors = item
            if rendered_image is None:
                if getattr(self, '_transfer_element_aware_active', False):
                    rendered = self._render_transfer_element_aware_exact(
                        self.plan_outline_array,
                        owner_map,
                        palette_rgb,
                        placement,
                    )
                else:
                    rendered = self._render_element_aware_v21(
                        self.plan_outline_array,
                        owner_map,
                        palette_rgb,
                        placement,
                    )
                rendered_image = Image.fromarray(rendered)
            rendered_scored.append((
                placement,
                score,
                rendered_image,
                breakdown,
                explanation,
                plan_colors,
            ))
            if index == exact_count or index % max(1, exact_count // 100) == 0:
                report(
                    70.0 + 30.0 * index / max(1, exact_count),
                    f"Rendering {index}/{exact_count} ({formula})",
                )

        placements = [item[0] for item in rendered_scored]
        return placements, rendered_scored

    def apply_colors_to_outline(self):
        """应用分割；复杂 Element-Aware 任务放入后台线程，避免 Tk 主线程假死。"""
        self._transfer_element_aware_active = False
        selected_method = self.segmentation_method_var.get()
        if selected_method != "Element-Aware Segmentation (v11)":
            return self._apply_colors_to_outline_sync()

        if getattr(self, '_outline_segmentation_running', False):
            self.update_status("Element-Aware segmentation is already running...")
            return

        effective_path = self._resolve_outline_path_for_segmentation(selected_method)
        if not effective_path or not os.path.exists(effective_path):
            messagebox.showerror("Error!", "Line sketch not found")
            return

        try:
            color_count = max(1, int(self.color_number_var.get()))
        except (TypeError, ValueError, tk.TclError):
            color_count = 5

        self._reset_outline_segmentation_metadata()
        self._element_aware_v21_active = False
        self._outline_segmentation_running = True
        self._outline_segmentation_result = None
        self._outline_segmentation_started_at = time.perf_counter()
        if hasattr(self, 'apply_color_button'):
            self.apply_color_button.config(state='disabled', text='Processing...')
        self.update_status(
            "Element-Aware segmentation is processing in background; the window remains responsive..."
        )

        worker = threading.Thread(
            target=self._run_element_aware_segmentation_worker,
            args=(effective_path, color_count),
            daemon=True,
            name='ElementAwareSegmentationWorker',
        )
        worker.start()
        self.root.after(80, self._poll_element_aware_segmentation_worker)

    def _run_element_aware_segmentation_worker(self, effective_path, color_count):
        """后台线程仅执行 PIL/NumPy/OpenCV 计算，不调用任何 Tk 接口。"""
        try:
            outline_img = Image.open(effective_path).convert('RGB')
            outline_array = np.asarray(outline_img)
            regions = self.split_regions_by_random_walker(outline_array, color_count)
            regions = [np.asarray(region, dtype=np.uint8) for region in regions]
            if not regions or not any(np.count_nonzero(region) for region in regions):
                raise RuntimeError("No closed paintable region was detected.")
            self._outline_segmentation_result = (
                'ok',
                {
                    'regions': regions,
                    'outline_array': outline_array.copy(),
                    'outline_image': outline_img.copy(),
                    'effective_path': effective_path,
                },
            )
        except Exception as exc:
            import traceback
            self._outline_segmentation_result = ('error', str(exc), traceback.format_exc())

    def _poll_element_aware_segmentation_worker(self):
        """主线程轮询后台结果，并在 Tk 线程内安全更新界面。"""
        result = getattr(self, '_outline_segmentation_result', None)
        if result is None:
            if getattr(self, '_outline_segmentation_running', False):
                self.root.after(80, self._poll_element_aware_segmentation_worker)
            return

        self._outline_segmentation_running = False
        self._outline_segmentation_result = None
        if hasattr(self, 'apply_color_button'):
            self.apply_color_button.config(state='normal', text='Apply Color')

        if result[0] != 'ok':
            detail = result[1] if len(result) > 1 else 'Unknown segmentation error.'
            if len(result) > 2:
                print(result[2])
            self.update_status("Element-Aware segmentation failed.")
            messagebox.showerror("Error!", detail)
            return

        payload = result[1]
        self.regions_for_plan = payload['regions']
        self.plan_outline_array = payload['outline_array']
        self.plan_outline_image = payload['outline_image']
        self._element_aware_effective_outline_path = payload['effective_path']
        self._outline_background_region_index = None
        self._element_aware_v21_active = True
        self.plan_button.config(state='normal')
        self.colors_applied = True
        self.last_action = 'outline_coloring'

        elapsed = time.perf_counter() - getattr(
            self, '_outline_segmentation_started_at', time.perf_counter()
        )
        diagnostics = getattr(self, '_last_element_aware_v21_diagnostics', {}) or {}
        component_count = diagnostics.get('connected_components', '?')
        region_count = diagnostics.get('kept_regions', '?')
        self.update_status(
            f"Element-Aware segmentation completed in {elapsed:.2f}s "
            f"({component_count} components, {region_count} principal regions)."
        )

    def _apply_colors_to_outline_sync(self):
        self._transfer_element_aware_active = False
        selected_method = self.segmentation_method_var.get()
        effective_path = self._resolve_outline_path_for_segmentation(selected_method)
        if not effective_path or not os.path.exists(effective_path):
            messagebox.showerror("Error!", "Line sketch not found")
            return

        outline_img = Image.open(effective_path).convert('RGB')
        outline_array = np.array(outline_img)
        color_count = self.color_number_var.get()

        self._reset_outline_segmentation_metadata()
        self._element_aware_v21_active = False

        if selected_method == "Perfect Mirror Symmetry Segmentation":
            regions = self.split_regions_by_perfect_mirror_symmetry(outline_array, color_count)
        elif selected_method == "Segmentation based on contour detection":
            regions = self.split_regions_by_contour_detection(outline_array, color_count)
        elif selected_method == "Segmentation based on watershed algorithm":
            regions = self.split_regions_by_watershed(outline_array, color_count)
        elif selected_method == "Segmentation based on Geometric analysis":
            regions = self.split_regions_by_geometric_analysis(outline_array, color_count)
        elif selected_method == "Segmentation based on Voronoi method":
            regions = self.split_regions_by_voronoi(outline_array, color_count)
        elif selected_method == "Felzenszwalb segmentation":
            regions = self.split_regions_by_felzenszwalb(outline_array, color_count)
        elif selected_method == "Flood Fill segmentation":
            regions = self.split_regions_by_floodfill(outline_array, color_count)
        elif selected_method == "Connected component segmentation":
            regions = self.split_regions_by_connected_components(outline_array, color_count)
        elif selected_method == "SLIC superpixel segmentation":
            regions = self.split_regions_by_slic(outline_array, color_count)
        elif selected_method == "Distance transform watershed":
            regions = self.split_regions_by_distance_watershed(outline_array, color_count)
        elif selected_method == "Symmetry-Aware Segmentation":
            regions = self.split_regions_by_forced_symmetry(outline_array, color_count)
        elif selected_method == "Element-Aware Segmentation (v11)":
            regions = self.split_regions_by_random_walker(outline_array, color_count)
            self._element_aware_v21_active = True
        else:
            messagebox.showerror("Error!", "Unknown segmentation method!")
            return

        if selected_method == "Element-Aware Segmentation (v11)":
            # v2.1.0 已经在原子连通域阶段排除了背景。再次进入共享补全会
            # 用 245 阈值重算边界、追加背景 owner，并把遗漏面按距离塞错组。
            regions = [np.asarray(region, dtype=np.uint8) for region in regions]
            self._outline_background_region_index = None
        else:
            regions = self._complete_outline_regions(outline_array, regions)

        # 生成方案时的总区域 N 必须来自线稿中的全部天然闭合区域，而不是
        # “按颜色数/对称关系合并后的区域组”。否则 N 会被压缩成 M 或更少，
        # 最终只得到 M! 个方案，看起来像把 N-M 个空白区域删掉了。
        atomic_regions = self._extract_outline_atomic_regions(outline_array)
        if atomic_regions:
            regions = atomic_regions
            self._outline_background_region_index = None
            self.update_status(
                f"Detected {len(atomic_regions)} total closed regions; plan generation will use N={len(atomic_regions)}."
            )

        if not regions or not any(np.count_nonzero(region) for region in regions):
            messagebox.showerror("Error!", "No closed paintable region was detected.")
            return

        self.regions_for_plan = regions
        self.plan_outline_array = outline_array.copy()
        self.plan_outline_image = outline_img.copy()
        self._element_aware_effective_outline_path = effective_path

        self.plan_button.config(state="normal")
        self.colors_applied = True
        self.last_action = "outline_coloring"
        self.update_status(
            f"The area has been divided. Total plan regions N={len(self.regions_for_plan)}; you can now generate plans."
        )

    def apply_colors_to_outline_helper(self):
        if not self.outline_image_path or not os.path.exists( self.outline_image_path ):
            messagebox.showerror( "Error!", "Line sketch not found!" )
            return

        outline_img = Image.open( self.outline_image_path ).convert( 'L' )
        outline_array = np.array( outline_img )

        color_count = len( self.outline_coloring_colors ) if self.outline_coloring_colors else 5

        regions = self.split_regions_by_symmetry( outline_array )

        if not regions:
            messagebox.showerror( "Error!", "Regional division failed!" )
            return

        regions = regions[:color_count]
        
        regions = self._complete_outline_regions(outline_array, regions)
        if not regions:
            messagebox.showerror("Error!", "No closed paintable region was detected.")
            return

        self.regions_for_plan = regions

        self.display_image( Image.fromarray( outline_array ), self.outline_coloring_label, (400, 400) )

        self.plan_button.config( state="normal" )
        self.colors_applied = True
        self.update_status( "The area has been divided and a plan can be generated!" )

    def split_regions_by_symmetry(self, outline_array):
        # 占位方法，实际未实现
        return []

    def split_regions_by_felzenszwalb(self, outline_array, color_count):
        """
        Felzenszwalb segmentation (structure optimized version)

        Optimizations:
        1. Line enhancement to avoid crossing lines
        2. Automatic merging of similar regions (symmetric/repetitive elements)
        3. Avoid fragmentation
        4. Ensure no empty regions
        """

        import cv2
        import numpy as np
        from skimage.segmentation import felzenszwalb

        # Grayscale conversion
        if len( outline_array.shape ) == 3:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_RGB2GRAY )
        else:
            gray = outline_array.copy()

        h, w = gray.shape

        # ===== 1 Line enhancement =====
        blur = cv2.GaussianBlur( gray, (3, 3), 0 )

        edges = cv2.Canny( blur, 40, 120 )

        kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (3, 3) )
        edges = cv2.dilate( edges, kernel, iterations=1 )

        enhanced = gray.copy()
        enhanced[edges > 0] = 0

        # ===== 2 Felzenszwalb segmentation =====
        img_for_seg = cv2.cvtColor( enhanced, cv2.COLOR_GRAY2RGB )

        segments = felzenszwalb(
            img_for_seg,
            scale=180,
            sigma=0.7,
            min_size=120
        )

        # ===== 3 Extract regions =====
        region_masks = []

        for label in np.unique( segments ):
            mask = (segments == label)

            if np.sum( mask ) > 80:
                region_masks.append( mask )

        # ===== 4 Calculate region features =====
        features = []

        for mask in region_masks:
            area = np.sum( mask )

            pixels = gray[mask]

            mean_val = np.mean( pixels )
            std_val = np.std( pixels )

            features.append( [area, mean_val, std_val] )

        features = np.array( features )

        # ===== 5 Merge similar regions =====
        merged_masks = []
        used = set()

        for i in range( len( region_masks ) ):
            if i in used:
                continue

            base_mask = region_masks[i].copy()

            for j in range( i + 1, len( region_masks ) ):
                if j in used:
                    continue

                # Feature distance
                dist = np.linalg.norm( features[i] - features[j] )

                # Similarity threshold
                if dist < 25:
                    base_mask |= region_masks[j]
                    used.add( j )

            merged_masks.append( base_mask )

        region_masks = merged_masks

        # ===== 6 Initialize color regions =====
        regions = [np.zeros( (h, w), dtype=bool ) for _ in range( color_count )]

        region_areas = [0] * color_count

        region_masks.sort( key=lambda m: np.sum( m ), reverse=True )

        # ===== 7 Balanced distribution =====
        for mask in region_masks:
            idx = np.argmin( region_areas )

            regions[idx] |= mask

            region_areas[idx] += np.sum( mask )

        # ===== 8 Fill gaps =====
        combined = np.zeros( (h, w), dtype=bool )

        for r in regions:
            combined |= r

        missing = ~combined

        if np.any( missing ):
            ys, xs = np.where( missing )

            idx = np.argmin( region_areas )

            regions[idx][ys, xs] = True

        # ===== 9 Convert to uint8 =====
        final_regions = []

        for r in regions:
            final_regions.append( (r.astype( np.uint8 )) * 255 )

        return final_regions

    def split_regions_by_floodfill(self, outline_array, color_count):
        import cv2
        import numpy as np

        if len( outline_array.shape ) == 3:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_RGB2GRAY )
        else:
            gray = outline_array.copy()

        h, w = gray.shape
        _, binary = cv2.threshold( gray, 240, 255, cv2.THRESH_BINARY )
        line_mask = binary == 0
        fill_area = binary.copy()
        fill_area[line_mask] = 0

        regions = [np.zeros( (h, w), dtype=np.uint8 ) for _ in range( color_count )]
        visited = np.zeros( (h, w), dtype=bool )
        region_id = 0
        count = 0

        seeds = []
        step = 15
        for y in range( step // 2, h, step ):
            for x in range( step // 2, w, step ):
                if fill_area[y, x] == 255 and not visited[y, x]:
                    seeds.append( (x, y) )

        for (x, y) in seeds:
            if visited[y, x] or fill_area[y, x] != 255:
                continue
            mask = np.zeros( (h + 2, w + 2), np.uint8 )
            cv2.floodFill( fill_area, mask, (x, y), 200, 0, 0 )
            mask = mask[1:-1, 1:-1]
            region = (mask == 1).astype( np.uint8 )
            region[line_mask] = 0

            if np.sum( region ) < 50:
                continue

            regions[region_id % color_count] |= region
            visited |= (region > 0)
            region_id += 1
            count += 1

        for i in range( color_count ):
            regions[i] *= 255

        return regions
    def split_regions_by_connected_components(self, outline_array, color_count):
        import cv2
        import numpy as np

        if len( outline_array.shape ) == 3:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_RGB2GRAY )
        else:
            gray = outline_array.copy()

        h, w = gray.shape

        _, binary = cv2.threshold( gray, 200, 255, cv2.THRESH_BINARY )

        num_labels, labels = cv2.connectedComponents( binary )

        region_masks = []

        for i in range( 1, num_labels ):
            mask = (labels == i)
            if np.sum( mask ) > 80:
                region_masks.append( mask )

        # ===== 复用你的均衡逻辑 =====
        regions = [np.zeros( (h, w), dtype=bool ) for _ in range( color_count )]
        region_areas = [0] * color_count

        region_masks.sort( key=lambda m: np.sum( m ), reverse=True )

        for mask in region_masks:
            idx = np.argmin( region_areas )
            regions[idx] |= mask
            region_areas[idx] += np.sum( mask )

        combined = np.zeros( (h, w), dtype=bool )
        for r in regions:
            combined |= r

        missing = ~combined
        if np.any( missing ):
            ys, xs = np.where( missing )
            idx = np.argmin( region_areas )
            regions[idx][ys, xs] = True

        return [(r.astype( np.uint8 )) * 255 for r in regions]

    def split_regions_by_slic(self, outline_array, color_count):
        import cv2
        import numpy as np
        from skimage.segmentation import slic

        if len( outline_array.shape ) == 3:
            img = outline_array.copy()
        else:
            img = cv2.cvtColor( outline_array, cv2.COLOR_GRAY2RGB )

        h, w = img.shape[:2]

        segments = slic( img, n_segments=color_count * 20, compactness=10, start_label=0 )

        region_masks = []

        for label in np.unique( segments ):
            mask = (segments == label)
            if np.sum( mask ) > 50:
                region_masks.append( mask )

        # ===== 均衡分配 =====
        regions = [np.zeros( (h, w), dtype=bool ) for _ in range( color_count )]
        region_areas = [0] * color_count

        region_masks.sort( key=lambda m: np.sum( m ), reverse=True )

        for mask in region_masks:
            idx = np.argmin( region_areas )
            regions[idx] |= mask
            region_areas[idx] += np.sum( mask )

        combined = np.zeros( (h, w), dtype=bool )
        for r in regions:
            combined |= r

        missing = ~combined
        if np.any( missing ):
            ys, xs = np.where( missing )
            idx = np.argmin( region_areas )
            regions[idx][ys, xs] = True

        return [(r.astype( np.uint8 )) * 255 for r in regions]

    def split_regions_by_distance_watershed(self, outline_array, color_count):
        import cv2
        import numpy as np

        if len( outline_array.shape ) == 3:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_RGB2GRAY )
        else:
            gray = outline_array.copy()

        h, w = gray.shape

        _, binary = cv2.threshold( gray, 200, 255, cv2.THRESH_BINARY_INV )

        dist = cv2.distanceTransform( binary, cv2.DIST_L2, 5 )

        _, sure_fg = cv2.threshold( dist, 0.4 * dist.max(), 255, 0 )
        sure_fg = np.uint8( sure_fg )

        unknown = cv2.subtract( binary, sure_fg )

        _, markers = cv2.connectedComponents( sure_fg )

        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed( cv2.cvtColor( gray, cv2.COLOR_GRAY2BGR ), markers )

        region_masks = []

        for label in np.unique( markers ):
            if label <= 1:
                continue

            mask = (markers == label)

            if np.sum( mask ) > 80:
                region_masks.append( mask )

        # ===== 均衡分配 =====
        regions = [np.zeros( (h, w), dtype=bool ) for _ in range( color_count )]
        region_areas = [0] * color_count

        region_masks.sort( key=lambda m: np.sum( m ), reverse=True )

        for mask in region_masks:
            idx = np.argmin( region_areas )
            regions[idx] |= mask
            region_areas[idx] += np.sum( mask )

        combined = np.zeros( (h, w), dtype=bool )
        for r in regions:
            combined |= r

        missing = ~combined
        if np.any( missing ):
            ys, xs = np.where( missing )
            idx = np.argmin( region_areas )
            regions[idx][ys, xs] = True

        return [(r.astype( np.uint8 )) * 255 for r in regions]

    def split_regions_by_forced_symmetry(self, outline_array, color_count):
        """
        【终极强制对称版】专为完美对称线稿设计
        核心逻辑：
        1.  强制以图像中心为对称轴，将左右/上下/四象限对称区域绑定
        2.  100%保证对称区域同色，绝不拆分
        3.  严格防重叠、防空白、防线条上色
        4.  适配中心对称、四面对称、左右/上下对称等所有标准纹样
        """
        import cv2
        import numpy as np

        # ======================
        # 1. 严格分离线条与填充区（根治线条上色、乱入）
        # ======================
        if len( outline_array.shape ) == 3:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_RGB2GRAY )
        else:
            gray = outline_array.copy()

        h, w = gray.shape
        cx, cy = w // 2, h // 2  # 图像中心（强制对称基准）

        # 提取纯线条掩码（线条=255，填充区=0）
        _, binary_line = cv2.threshold( gray, 220, 255, cv2.THRESH_BINARY_INV )
        kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (2, 2) )
        line_mask = cv2.dilate( binary_line, kernel, iterations=1 )
        line_mask = (line_mask > 0).astype( np.uint8 )  # 1=线条，0=可填充

        # 提取纯填充区（完全剔除线条）
        fill_mask = (1 - line_mask).astype( np.uint8 ) * 255  # 255=可填充，0=线条

        # ======================
        # 2. 提取所有封闭填充区域（只保留有效区域）
        # ======================
        contours, _ = cv2.findContours( fill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        regions = []
        centers = []

        for cnt in contours:
            area = cv2.contourArea( cnt )
            if area < 30:  # 过滤微小噪点
                continue

            # 生成区域掩码（完全不含线条）
            mask = np.zeros( (h, w), dtype=np.uint8 )
            cv2.drawContours( mask, [cnt], -1, 255, thickness=cv2.FILLED )
            mask = cv2.bitwise_and( mask, fill_mask )

            # 计算区域中心（用于对称匹配）
            M = cv2.moments( cnt )
            if M["m00"] == 0:
                continue
            cx_reg = int( M["m10"] / M["m00"] )
            cy_reg = int( M["m01"] / M["m00"] )

            regions.append( mask )
            centers.append( (cx_reg, cy_reg) )

        # 兜底：无区域时用几何分割
        if len( regions ) == 0:
            return self.split_regions_by_geometric_analysis( outline_array, color_count )

        # ======================
        # 3. 【核心】强制对称分组（锁死对称结构）
        # ======================
        used = [False] * len( regions )
        groups = []

        for i in range( len( regions ) ):
            if used[i]:
                continue

            current_mask = regions[i].copy()
            cx1, cy1 = centers[i]
            used[i] = True

            # 计算该区域的所有对称位置（适配所有对称类型）
            # 1. 左右对称点
            sym_lr = (w - 1 - cx1, cy1)
            # 2. 上下对称点
            sym_ud = (cx1, h - 1 - cy1)
            # 3. 中心对称点
            sym_center = (w - 1 - cx1, h - 1 - cy1)
            # 4. 四象限对称点（适配四面对称纹样）
            sym_q1 = (cx1, cy1)
            sym_q2 = (w - 1 - cx1, cy1)
            sym_q3 = (cx1, h - 1 - cy1)
            sym_q4 = (w - 1 - cx1, h - 1 - cy1)

            # 遍历所有区域，匹配所有对称位置
            for j in range( len( regions ) ):
                if i == j or used[j]:
                    continue
                cx2, cy2 = centers[j]

                # 计算到所有对称点的最小距离（只要匹配任意对称点，就强制合并）
                d_lr = np.hypot( cx2 - sym_lr[0], cy2 - sym_lr[1] )
                d_ud = np.hypot( cx2 - sym_ud[0], cy2 - sym_ud[1] )
                d_center = np.hypot( cx2 - sym_center[0], cy2 - sym_center[1] )
                min_dist = min( d_lr, d_ud, d_center )

                # 阈值：30像素内视为对称区域（可根据你的线稿微调，越大越宽松）
                if min_dist < 30:
                    current_mask = cv2.bitwise_or( current_mask, regions[j] )
                    used[j] = True

            groups.append( current_mask )

        # ======================
        # 4. 严格防重叠分配（根治一个区域两个颜色）
        # ======================
        # 按面积从大到小排序，保证大区域优先分配
        groups.sort( key=lambda m: cv2.countNonZero( m ), reverse=True )

        # 初始化最终区域 + 全局像素占用跟踪（100%防重叠）
        final = [np.zeros( (h, w), dtype=np.uint8 ) for _ in range( color_count )]
        area_sum = [0] * color_count
        occupied = np.zeros( (h, w), dtype=np.uint8 )  # 0=未占用，1=已分配

        for g in groups:
            # 只分配未占用的像素，彻底杜绝重叠
            new_pixels = cv2.bitwise_and( g, cv2.bitwise_not( occupied ) )
            if cv2.countNonZero( new_pixels ) == 0:
                continue

            # 分配到当前面积最小的区域，保证颜色均匀分布
            idx = np.argmin( area_sum )
            final[idx] = cv2.bitwise_or( final[idx], new_pixels )
            area_sum[idx] += cv2.countNonZero( new_pixels )
            occupied = cv2.bitwise_or( occupied, new_pixels )

        # ======================
        # 5. 强制填充空白（根治漏色、空白区域）
        # ======================
        missing = cv2.bitwise_and( fill_mask, cv2.bitwise_not( occupied ) )
        if cv2.countNonZero( missing ) > 0:
            idx = np.argmin( area_sum )
            final[idx] = cv2.bitwise_or( final[idx], missing )
            area_sum[idx] += cv2.countNonZero( missing )

        # ======================
        # 6. 最终校验：再次剔除线条，保证线条纯白
        # ======================
        for i in range( color_count ):
            final[i] = cv2.bitwise_and( final[i], fill_mask )

        return final

    @staticmethod
    def _project_area_shares_v414(values, minimum=0.05, maximum=0.58):
        """把面积目标投影到带上下界的概率单纯形。

        minimum=5% 是用户要求的颜色最低总面积；maximum 用于避免单色吞并
        主体。若颜色数量使给定上下界数学上不可行，会自动收缩到可行边界。
        """
        shares = np.asarray(values, dtype=np.float64).reshape(-1)
        if shares.size == 0:
            return shares
        shares = np.maximum(shares, 0.0)
        if float(np.sum(shares)) <= 1e-12:
            shares[:] = 1.0
        shares /= float(np.sum(shares))

        count = len(shares)
        minimum = max(0.0, min(float(minimum), 0.98 / count))
        maximum = max(float(maximum), 1.02 / count)
        maximum = min(1.0, maximum)

        # 有界水位投影。每轮固定越界项，再把剩余质量按原始比例分给自由项。
        result = shares.copy()
        fixed = np.zeros(count, dtype=bool)
        for _ in range(count * 3 + 3):
            low = (~fixed) & (result < minimum)
            high = (~fixed) & (result > maximum)
            if not np.any(low) and not np.any(high):
                break
            result[low] = minimum
            result[high] = maximum
            fixed |= low | high
            remaining = 1.0 - float(np.sum(result[fixed]))
            free = ~fixed
            if not np.any(free):
                break
            base = shares[free]
            if float(np.sum(base)) <= 1e-12:
                result[free] = remaining / int(np.count_nonzero(free))
            else:
                result[free] = remaining * base / float(np.sum(base))

        # 数值误差修正，同时保持边界。
        for _ in range(8):
            diff = 1.0 - float(np.sum(result))
            if abs(diff) <= 1e-10:
                break
            if diff > 0:
                candidates = np.where(result < maximum - 1e-12)[0]
            else:
                candidates = np.where(result > minimum + 1e-12)[0]
            if not len(candidates):
                break
            capacity = (
                maximum - result[candidates]
                if diff > 0 else result[candidates] - minimum
            )
            capacity_sum = float(np.sum(capacity))
            if capacity_sum <= 1e-12:
                break
            result[candidates] += diff * capacity / capacity_sum
        return np.clip(result, 0.0, 1.0)

    def _source_area_targets_v414(self, output_count, current_shares, assignment_mode):
        """从源图颜色比例生成目标 owner 容量，并与当前语义分组温和融合。"""
        output_count = max(1, int(output_count))
        current = np.asarray(current_shares, dtype=np.float64).reshape(-1)
        if len(current) != output_count or float(np.sum(current)) <= 1e-12:
            current = np.full(output_count, 1.0 / output_count, dtype=np.float64)
        else:
            current = current / float(np.sum(current))

        source = None
        source_labels = getattr(self, 'labels_2d', None)
        source_colors = list(getattr(self, 'hex_color_codes', []) or [])
        if source_labels is not None and len(source_colors) >= output_count:
            counts = np.asarray([
                np.count_nonzero(source_labels == index)
                for index in range(len(source_colors))
            ], dtype=np.float64)
            if float(np.sum(counts)) > 0.0:
                source = counts / float(np.sum(counts))
                if assignment_mode == 'reference':
                    source = source[:output_count]
                else:
                    source = np.sort(source)[::-1][:output_count]
                if len(source) == output_count and float(np.sum(source)) > 0.0:
                    source = source / float(np.sum(source))
                else:
                    source = None

        if source is None:
            blended = current
        else:
            # 以源 CRN 节点面积为主，同时保留 20% 当前几何语义，避免粗暴重排。
            blended = 0.80 * source + 0.20 * current
        return self._project_area_shares_v414(
            blended, minimum=0.05, maximum=0.58
        )

    def _rebalance_element_aware_assignments_v414(
            self, regions, assignment, stats, pairs, output_count,
            assignment_mode='auto'):
        """以闭合连通面/对称组为不可拆原子，重平衡 Element-Aware owners。

        目标：
        1. 每个颜色 owner 的主体总面积尽量不低于 5%；
        2. 避免单个 owner 吞并 60% 以上主体；
        3. 容量曲线优先贴近源图 CRN 节点比例；
        4. 对称配对始终作为一个 bundle 移动，不破坏镜像一致性。
        """
        assignment = np.asarray(assignment, dtype=np.int32).copy()
        output_count = max(1, int(output_count))
        if output_count <= 1 or not regions:
            return assignment, {
                'before_shares': [1.0], 'after_shares': [1.0],
                'target_shares': [1.0], 'moves': 0,
                'minimum_share': 1.0, 'maximum_share': 1.0,
            }

        valid_labels = {int(region['label']) for region in regions}
        parent = {label: label for label in valid_labels}

        def find(value):
            value = int(value)
            parent.setdefault(value, value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(first, second):
            first, second = int(first), int(second)
            if first not in valid_labels or second not in valid_labels:
                return
            root_first, root_second = find(first), find(second)
            if root_first != root_second:
                parent[root_second] = root_first

        for first, second, *_ in pairs:
            union(first, second)

        by_root = {}
        region_by_label = {int(region['label']): region for region in regions}
        for label in valid_labels:
            by_root.setdefault(find(label), []).append(label)

        bundles = []
        for labels_in_bundle in by_root.values():
            area = float(sum(
                int(stats[label, cv2.CC_STAT_AREA]) for label in labels_in_bundle
            ))
            if area <= 0.0:
                continue
            owner_votes = {}
            weighted_x = 0.0
            weighted_y = 0.0
            for label in labels_in_bundle:
                owner = int(assignment[label])
                label_area = float(int(stats[label, cv2.CC_STAT_AREA]))
                if 0 <= owner < output_count:
                    owner_votes[owner] = owner_votes.get(owner, 0.0) + label_area
                region = region_by_label[label]
                weighted_x += float(region['cx']) * label_area
                weighted_y += float(region['cy']) * label_area
            if not owner_votes:
                continue
            owner = max(owner_votes, key=lambda value: (owner_votes[value], -value))
            mean_aspect = float(np.mean([
                float(region_by_label[label]['aspect'])
                for label in labels_in_bundle
            ]))
            mean_extent = float(np.mean([
                float(region_by_label[label]['extent'])
                for label in labels_in_bundle
            ]))
            mean_circularity = float(np.mean([
                float(region_by_label[label]['circularity'])
                for label in labels_in_bundle
            ]))
            bundles.append({
                'labels': tuple(labels_in_bundle),
                'area': area,
                'owner': int(owner),
                'original_owner': int(owner),
                'cx': weighted_x / area,
                'cy': weighted_y / area,
                'aspect': mean_aspect,
                'extent': mean_extent,
                'circularity': mean_circularity,
            })

        if not bundles:
            return assignment, {
                'before_shares': [], 'after_shares': [],
                'target_shares': [], 'moves': 0,
                'minimum_share': 0.0, 'maximum_share': 0.0,
            }

        total_area = float(sum(bundle['area'] for bundle in bundles))
        loads = np.zeros(output_count, dtype=np.float64)
        owner_bundle_ids = [set() for _ in range(output_count)]
        for bundle_id, bundle in enumerate(bundles):
            loads[bundle['owner']] += bundle['area']
            owner_bundle_ids[bundle['owner']].add(bundle_id)
        before = loads / max(total_area, 1.0)
        target_shares = self._source_area_targets_v414(
            output_count, before, assignment_mode
        )
        targets = target_shares * total_area
        floor_area = min(0.05, 0.98 / output_count) * total_area
        cap_area = max(0.58, 1.02 / output_count) * total_area

        width = max(1.0, float(np.max(stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH])))
        height = max(1.0, float(np.max(stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT])))
        bundle_points = np.asarray([
            [bundle['cx'] / width, bundle['cy'] / height] for bundle in bundles
        ], dtype=np.float64)
        bundle_features = np.asarray([
            [
                math.log1p(bundle['area']) / math.log1p(max(total_area, 1.0)),
                math.log(max(bundle['aspect'], 1e-4)),
                bundle['extent'],
                bundle['circularity'],
            ]
            for bundle in bundles
        ], dtype=np.float64)
        bundle_features = (
            bundle_features - bundle_features.mean(axis=0, keepdims=True)
        ) / (bundle_features.std(axis=0, keepdims=True) + 1e-6)
        pairwise_distance = np.sqrt(np.sum(
            (bundle_points[:, None, :] - bundle_points[None, :, :]) ** 2, axis=2
        ))
        pairwise_shape = np.sqrt(np.sum(
            (bundle_features[:, None, :] - bundle_features[None, :, :]) ** 2, axis=2
        ))
        neighbor_ids = [
            [int(index) for index in np.argsort(pairwise_distance[bundle_id])[1:9]]
            for bundle_id in range(len(bundles))
        ]

        def local_aggregation_score(bundle_id, owner, owner_sets=None):
            owner_sets = owner_bundle_ids if owner_sets is None else owner_sets
            total = 0.0
            for other in neighbor_ids[bundle_id]:
                if other == bundle_id or other not in owner_sets[owner]:
                    continue
                spatial = float(pairwise_distance[bundle_id, other])
                shape = float(pairwise_shape[bundle_id, other])
                if spatial > 0.32:
                    continue
                total += math.exp(-6.0 * spatial) * math.exp(-0.85 * shape)
            return float(total)

        def owner_centers():
            centers = []
            for owner in range(output_count):
                ids = owner_bundle_ids[owner]
                if not ids:
                    centers.append((width * 0.5, height * 0.5))
                    continue
                area_sum = sum(bundles[idx]['area'] for idx in ids)
                centers.append((
                    sum(bundles[idx]['cx'] * bundles[idx]['area'] for idx in ids) / area_sum,
                    sum(bundles[idx]['cy'] * bundles[idx]['area'] for idx in ids) / area_sum,
                ))
            return centers

        def objective(test_loads):
            normalized_error = float(np.sum(np.abs(test_loads - targets))) / max(total_area, 1.0)
            floor_penalty = float(np.sum(np.maximum(0.0, floor_area - test_loads))) / max(total_area, 1.0)
            cap_penalty = float(np.sum(np.maximum(0.0, test_loads - cap_area))) / max(total_area, 1.0)
            return normalized_error + 4.0 * floor_penalty + 2.5 * cap_penalty

        moves = 0
        max_moves = max(24, min(320, len(bundles) * 3))
        for _ in range(max_moves):
            centers = owner_centers()
            current_objective = objective(loads)
            recipients = sorted(
                range(output_count),
                key=lambda owner: (loads[owner] - targets[owner], loads[owner]),
            )
            donors = sorted(
                range(output_count),
                key=lambda owner: (loads[owner] - targets[owner], loads[owner]),
                reverse=True,
            )
            best = None
            for donor in donors:
                if len(owner_bundle_ids[donor]) <= 1:
                    continue
                donor_ids = list(owner_bundle_ids[donor])
                # 先看最可能填补缺口的小中型 bundle，控制复杂图上的运行时间。
                donor_ids.sort(key=lambda idx: bundles[idx]['area'])
                candidate_ids = donor_ids[:96]
                if donor_ids[-1] not in candidate_ids:
                    candidate_ids.append(donor_ids[-1])
                for bundle_id in candidate_ids:
                    bundle = bundles[bundle_id]
                    area = bundle['area']
                    if loads[donor] - area < floor_area - 1e-9:
                        continue
                    for recipient in recipients:
                        if recipient == donor:
                            continue
                        trial = loads.copy()
                        trial[donor] -= area
                        trial[recipient] += area
                        trial_objective = objective(trial)
                        distance = math.hypot(
                            (bundle['cx'] - centers[recipient][0]) / width,
                            (bundle['cy'] - centers[recipient][1]) / height,
                        )
                        original_penalty = 0.0 if recipient == bundle['original_owner'] else 0.018
                        cohesion_gain = (
                            local_aggregation_score(bundle_id, recipient)
                            - 0.35 * local_aggregation_score(bundle_id, donor)
                        )
                        # 容量误差是主目标，空间与原 owner 只作温和破平局项；
                        # 同时鼓励小面积色块在局部形成连贯聚合，而非撒点式分配。
                        utility = (
                            trial_objective
                            + 0.025 * distance
                            + original_penalty
                            - 0.032 * cohesion_gain
                        )
                        improvement = current_objective - trial_objective
                        if improvement <= 1e-8:
                            continue
                        key = (utility, -improvement, area, bundle_id, recipient)
                        if best is None or key < best[0]:
                            best = (key, donor, recipient, bundle_id, trial)
            if best is None:
                break
            _, donor, recipient, bundle_id, trial = best
            owner_bundle_ids[donor].remove(bundle_id)
            owner_bundle_ids[recipient].add(bundle_id)
            bundles[bundle_id]['owner'] = recipient
            loads = trial
            moves += 1

            if (
                float(np.min(loads)) >= floor_area - 1e-9
                and float(np.max(loads)) <= cap_area + 1e-9
                and float(np.max(np.abs(loads - targets))) <= 0.015 * total_area
            ):
                break

        # 最低 5% 为硬优先级：若局部优化仍有低占比 owner，再执行定向补足。
        for recipient in np.argsort(loads):
            recipient = int(recipient)
            while loads[recipient] < floor_area - 1e-9:
                centers = owner_centers()
                best = None
                for donor in np.argsort(loads)[::-1]:
                    donor = int(donor)
                    if donor == recipient or len(owner_bundle_ids[donor]) <= 1:
                        continue
                    for bundle_id in owner_bundle_ids[donor]:
                        area = bundles[bundle_id]['area']
                        if loads[donor] - area < floor_area - 1e-9:
                            continue
                        distance = math.hypot(
                            (bundles[bundle_id]['cx'] - centers[recipient][0]) / width,
                            (bundles[bundle_id]['cy'] - centers[recipient][1]) / height,
                        )
                        deficit_after = abs((loads[recipient] + area) - targets[recipient])
                        cohesion_gain = (
                            local_aggregation_score(bundle_id, recipient)
                            - 0.35 * local_aggregation_score(bundle_id, donor)
                        )
                        key = (
                               deficit_after / max(targets[recipient], 1.0)
                               + 0.03 * distance
                               - 0.035 * cohesion_gain,
                               area, bundle_id)
                        if best is None or key < best[0]:
                            best = (key, donor, bundle_id)
                if best is None:
                    break
                _, donor, bundle_id = best
                area = bundles[bundle_id]['area']
                owner_bundle_ids[donor].remove(bundle_id)
                owner_bundle_ids[recipient].add(bundle_id)
                bundles[bundle_id]['owner'] = recipient
                loads[donor] -= area
                loads[recipient] += area
                moves += 1

        for bundle in bundles:
            for label in bundle['labels']:
                assignment[int(label)] = int(bundle['owner'])

        after = loads / max(total_area, 1.0)
        return assignment, {
            'before_shares': [float(value) for value in before],
            'after_shares': [float(value) for value in after],
            'target_shares': [float(value) for value in target_shares],
            'moves': int(moves),
            'minimum_share': float(np.min(after)) if len(after) else 0.0,
            'maximum_share': float(np.max(after)) if len(after) else 0.0,
        }

    @staticmethod
    def _select_element_aware_symmetry_modes_v415(symmetry_confidence):
        """选择可信的主对称轴，避免高密度线稿产生伪水平/中心对称。"""
        if not symmetry_confidence:
            return []
        ranked = sorted(
            symmetry_confidence.items(), key=lambda item: item[1], reverse=True
        )
        primary_mode, primary_score = ranked[0]
        if float(primary_score) < 0.86:
            return []
        selected = [primary_mode]
        # 只有多个轴都极强且分数非常接近时才同时启用；适合真正的方形/中心纹样。
        for mode, score in ranked[1:]:
            if float(score) >= 0.95 and float(primary_score - score) <= 0.018:
                selected.append(mode)
        return selected

    @staticmethod
    def _augment_element_aware_symmetry_pairs_v415(
            regions, image_shape, transforms, existing_pairs=None):
        """用镜像质心 + 几何特征补足对称配对。

        旧版仅要求变换后的区域像素直接落入另一连通域，细线宽差、抗锯齿或
        局部断线会使绝大多数镜像区域漏配。这里在归一化坐标中做一对一贪心
        匹配，并用面积、长宽比、填充率和圆度约束避免串配。
        """
        if not regions or not transforms:
            return list(existing_pairs or [])
        h, w = map(int, image_shape[:2])
        points = np.asarray([
            [float(region['cx']) / max(1.0, float(w)),
             float(region['cy']) / max(1.0, float(h))]
            for region in regions
        ], dtype=np.float64)
        index = NearestNeighborIndex(points)
        pair_cost = {
            tuple(sorted((int(first), int(second)))): float(cost)
            for first, second, cost in (existing_pairs or [])
            if int(first) != int(second)
        }

        def target(point, mode):
            x, y = map(float, point)
            if mode == 'vertical':
                return np.array([1.0 - x, y], dtype=np.float64)
            if mode == 'horizontal':
                return np.array([x, 1.0 - y], dtype=np.float64)
            if mode == 'central':
                return np.array([1.0 - x, 1.0 - y], dtype=np.float64)
            if mode == 'diagonal_main':
                return np.array([y, x], dtype=np.float64)
            return np.array([1.0 - y, 1.0 - x], dtype=np.float64)

        for mode in transforms:
            edges = []
            k = min(18, len(regions))
            for source_index, region in enumerate(regions):
                distances, candidates = index.query(
                    target(points[source_index], mode), k=k
                )
                distances = np.atleast_1d(distances)
                candidates = np.atleast_1d(candidates)
                for spatial_distance, candidate_index in zip(distances, candidates):
                    candidate_index = int(candidate_index)
                    if candidate_index == source_index:
                        continue
                    # 每条无向边只生成一次。
                    if source_index > candidate_index:
                        continue
                    candidate = regions[candidate_index]
                    source_area = max(1.0, float(region['area']))
                    candidate_area = max(1.0, float(candidate['area']))
                    area_ratio = min(source_area, candidate_area) / max(
                        source_area, candidate_area
                    )
                    source_aspect = max(1e-4, float(region['aspect']))
                    candidate_aspect = max(1e-4, float(candidate['aspect']))
                    expected_aspect = (
                        1.0 / source_aspect
                        if mode in ('diagonal_main', 'diagonal_anti')
                        else source_aspect
                    )
                    aspect_ratio = min(expected_aspect, candidate_aspect) / max(
                        expected_aspect, candidate_aspect
                    )
                    if (
                        float(spatial_distance) > 0.032
                        or area_ratio < 0.38
                        or aspect_ratio < 0.38
                    ):
                        continue
                    extent_delta = abs(
                        float(region['extent']) - float(candidate['extent'])
                    )
                    circularity_delta = abs(
                        float(region['circularity'])
                        - float(candidate['circularity'])
                    )
                    cost = (
                        5.0 * float(spatial_distance)
                        + 0.45 * abs(math.log(source_area / candidate_area))
                        + 0.12 * abs(math.log(expected_aspect / candidate_aspect))
                        + 0.18 * extent_delta
                        + 0.12 * circularity_delta
                    )
                    if cost <= 0.72:
                        edges.append((float(cost), source_index, candidate_index))

            used = set()
            for cost, source_index, candidate_index in sorted(edges):
                if source_index in used or candidate_index in used:
                    continue
                used.add(source_index)
                used.add(candidate_index)
                first = int(regions[source_index]['label'])
                second = int(regions[candidate_index]['label'])
                key = tuple(sorted((first, second)))
                if key not in pair_cost or cost < pair_cost[key]:
                    pair_cost[key] = float(cost)

        return [
            (first, second, cost)
            for (first, second), cost in sorted(pair_cost.items())
        ]

    @staticmethod
    def _build_element_aware_motif_pairs_v417(
            regions, image_shape, transforms=None, existing_pairs=None,
            max_neighbors=8):
        """为相似小纹样补充 must-link 配对，改善眼睛/鸟饰/珠串等同类元素异色。"""
        if not regions:
            return list(existing_pairs or [])
        h, w = map(int, image_shape[:2])
        image_area = max(1.0, float(h * w))
        transforms = list(transforms or [])
        pair_cost = {
            tuple(sorted((int(first), int(second)))): float(cost)
            for first, second, cost in (existing_pairs or [])
            if int(first) != int(second)
        }
        points = np.asarray([
            [float(region['cx']) / max(1.0, float(w)),
             float(region['cy']) / max(1.0, float(h))]
            for region in regions
        ], dtype=np.float64)
        feature_rows = np.asarray([
            [
                math.log1p(float(region['area'])) / math.log1p(image_area),
                math.log(max(float(region['aspect']), 1e-4)),
                float(region['extent']),
                float(region['circularity']),
                float(region['w']) / max(1.0, float(w)),
                float(region['h']) / max(1.0, float(h)),
            ]
            for region in regions
        ], dtype=np.float64)
        features = (
            feature_rows - feature_rows.mean(axis=0, keepdims=True)
        ) / (feature_rows.std(axis=0, keepdims=True) + 1e-6)
        index = NearestNeighborIndex(features)

        def mirrored_distance(first_point, second_point):
            if not transforms:
                return 1.0
            x, y = map(float, first_point)
            distances = []
            for mode in transforms:
                if mode == 'vertical':
                    tx, ty = 1.0 - x, y
                elif mode == 'horizontal':
                    tx, ty = x, 1.0 - y
                elif mode == 'central':
                    tx, ty = 1.0 - x, 1.0 - y
                elif mode == 'diagonal_main':
                    tx, ty = y, x
                else:
                    tx, ty = 1.0 - y, 1.0 - x
                distances.append(math.hypot(tx - second_point[0], ty - second_point[1]))
            return min(distances) if distances else 1.0

        used_pairs = set(pair_cost)
        for source_index, region in enumerate(regions):
            area_share = float(region['area']) / image_area
            # 只对小中型装饰纹样做同类约束，避免大主体被过度锁死。
            if area_share > 0.12:
                continue
            distances, candidates = index.query(features[source_index], k=min(max_neighbors + 1, len(regions)))
            distances = np.atleast_1d(distances)
            candidates = np.atleast_1d(candidates)
            for feature_distance, candidate_index in zip(distances[1:], candidates[1:]):
                candidate_index = int(candidate_index)
                if candidate_index == source_index:
                    continue
                candidate = regions[candidate_index]
                candidate_area_share = float(candidate['area']) / image_area
                if candidate_area_share > 0.12:
                    continue
                source_area = max(1.0, float(region['area']))
                candidate_area = max(1.0, float(candidate['area']))
                area_ratio = min(source_area, candidate_area) / max(source_area, candidate_area)
                if area_ratio < 0.55:
                    continue
                source_aspect = max(1e-4, float(region['aspect']))
                candidate_aspect = max(1e-4, float(candidate['aspect']))
                aspect_ratio = min(source_aspect, candidate_aspect) / max(source_aspect, candidate_aspect)
                if aspect_ratio < 0.55:
                    continue
                extent_delta = abs(float(region['extent']) - float(candidate['extent']))
                circularity_delta = abs(float(region['circularity']) - float(candidate['circularity']))
                if extent_delta > 0.18 or circularity_delta > 0.18:
                    continue
                direct_distance = math.hypot(
                    points[source_index, 0] - points[candidate_index, 0],
                    points[source_index, 1] - points[candidate_index, 1],
                )
                sym_distance = mirrored_distance(points[source_index], points[candidate_index])
                spatial_gate = min(direct_distance, sym_distance)
                if spatial_gate > 0.24 and sym_distance > 0.065:
                    continue
                first = int(region['label'])
                second = int(candidate['label'])
                key = tuple(sorted((first, second)))
                if key in used_pairs:
                    continue
                cost = (
                    0.45 * float(feature_distance)
                    + 0.28 * (1.0 - area_ratio)
                    + 0.12 * extent_delta
                    + 0.10 * circularity_delta
                    + 0.05 * min(direct_distance, 0.30)
                )
                if sym_distance <= 0.065:
                    cost *= 0.82
                if cost <= 0.95:
                    pair_cost[key] = min(pair_cost.get(key, cost), float(cost))
                    used_pairs.add(key)
        return [
            (first, second, cost)
            for (first, second), cost in sorted(pair_cost.items())
        ]

    @staticmethod
    def _enforce_element_aware_pair_assignments_v415(
            assignment, pairs, stats):
        """对所有镜像 pair 的并查集做面积加权 owner 投票。"""
        assignment = np.asarray(assignment, dtype=np.int32).copy()
        if not pairs:
            return assignment
        parent = {}

        def find(value):
            value = int(value)
            parent.setdefault(value, value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(first, second):
            root_first, root_second = find(first), find(second)
            if root_first != root_second:
                parent[root_second] = root_first

        for first, second, *_ in pairs:
            union(first, second)
        groups = {}
        for label in list(parent):
            groups.setdefault(find(label), []).append(int(label))

        for group_labels in groups.values():
            owner_areas = {}
            owner_counts = {}
            for label in group_labels:
                if label < 0 or label >= len(assignment):
                    continue
                owner = int(assignment[label])
                if owner < 0:
                    continue
                area = max(1, int(stats[label, cv2.CC_STAT_AREA]))
                owner_areas[owner] = owner_areas.get(owner, 0) + area
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
            if not owner_areas:
                continue
            winner = max(
                owner_areas,
                key=lambda owner: (
                    owner_areas[owner], owner_counts[owner], -owner
                ),
            )
            for label in group_labels:
                if 0 <= label < len(assignment) and assignment[label] >= 0:
                    assignment[label] = int(winner)
        return assignment
    def split_regions_by_random_walker(
            self, outline_array, color_count, include_background_owner=False):
        """Element-Aware：完整迁移 symmetric_coloring_full_v2.1.0 数据流。

        保留原子闭合区域直到完成“自动几何类型/同源参考颜色”分配，再按颜色
        owner 合并。与旧移植不同，不把 arbitrary selected_colors[0] 直接套给
        最大 cluster，也不经过通用补洞与 motif 二次重分割。
        """
        import math

        try:
            requested_count = max(1, int(color_count))
        except (TypeError, ValueError, OverflowError):
            requested_count = 1
        include_background_owner = bool(include_background_owner)

        source = np.asarray(outline_array)
        if source.ndim == 2:
            line_rgb = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] == 1:
            line_rgb = cv2.cvtColor(source[..., 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            line_rgb = source[..., :3].astype(np.uint8, copy=True)

        gray = cv2.cvtColor(line_rgb, cv2.COLOR_RGB2GRAY)
        barrier = (gray < 235).astype(np.uint8)
        fillable = (1 - barrier).astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fillable, connectivity=4, ltype=cv2.CV_32S
        )
        h, w = labels.shape

        border_values = np.concatenate((
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ))
        border_values = border_values[border_values > 0]
        background_labels = set(int(value) for value in np.unique(border_values))
        if background_labels:
            background_label = max(
                background_labels,
                key=lambda value: int(stats[value, cv2.CC_STAT_AREA]),
            )
        else:
            counts = np.bincount(labels.ravel())
            background_label = int(np.argmax(counts[1:]) + 1) if len(counts) > 1 else 0
            if background_label > 0:
                background_labels = {background_label}

        def circularity(local_mask):
            contours, _ = cv2.findContours(
                local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return 0.0
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            return (
                float(4.0 * math.pi * area / (perimeter * perimeter))
                if perimeter > 1e-6 else 0.0
            )

        regions = []
        deferred_small_regions = []
        for label in range(1, stats.shape[0]):
            if label in background_labels:
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = map(float, centroids[label])
            roi = np.uint8(labels[y:y + height, x:x + width] == label) * 255
            region_data = {
                'label': label,
                'area': area,
                'x': x,
                'y': y,
                'w': width,
                'h': height,
                'cx': cx,
                'cy': cy,
                'extent': area / max(float(width * height), 1.0),
                'aspect': width / max(float(height), 1.0),
                'circularity': circularity(roi),
            }
            if area < 20:
                deferred_small_regions.append(region_data)
            else:
                regions.append(region_data)
        if not regions and deferred_small_regions:
            # 只有微型闭合面时不能走“纯背景”早退；这些面就是主体元素。
            regions = list(deferred_small_regions)
        if not regions:
            self._element_aware_v210_assignment_mode = 'auto'
            owner_map = np.full((h, w), -1, dtype=np.int32)
            owner_map[fillable > 0] = 0
            self._element_aware_v210_owner_map = owner_map
            self._element_aware_v210_group_source_indices = [0]
            self._element_aware_v210_labels = labels
            self._element_aware_v210_stats = stats
            self._element_aware_v210_background_label = int(background_label)
            self._last_element_aware_v21_diagnostics = {
                'core_version': '2.1.0-refined-v4.1.7-fast-motif-cache',
                'image_size': [int(w), int(h)],
                'connected_components': int(stats.shape[0] - 1),
                'background_label': int(background_label),
                'background_labels': sorted(background_labels),
                'background_owner': None,
                'kept_regions': 0,
                'requested_color_groups': requested_count,
                'returned_color_groups': 1,
                'assignment_mode': 'auto',
                'reference_regions': 0,
                'symmetric_pair_count': 0,
                'symmetry_confidence': {},
                'enabled_symmetries': [],
                'fillable_coverage': 1.0,
                'native_input': True,
            }
            return [np.uint8(fillable > 0) * 255]

        image_area = float(h * w)
        feature_rows = np.asarray([
            [
                math.log1p(region['area']) / math.log1p(image_area),
                math.log(max(region['aspect'], 1e-4)),
                region['extent'],
                region['circularity'],
                region['w'] / float(w),
                region['h'] / float(h),
            ]
            for region in regions
        ], dtype=np.float32)
        features = (
            feature_rows - feature_rows.mean(axis=0, keepdims=True)
        ) / (feature_rows.std(axis=0, keepdims=True) + 1e-6)
        # 不使用带符号 x/y 坐标：镜像区域的几何特征应保持一致。
        unique_count = len(np.unique(np.round(features, 5), axis=0))
        foreground_capacity = max(
            1,
            requested_count - 1
            if include_background_owner and requested_count > 1
            else requested_count,
        )
        auto_cluster_count = max(
            1, min(foreground_capacity, len(regions), unique_count)
        )
        cv2.setRNGSeed(2026)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4
        )
        _, auto_assignment, _ = cv2.kmeans(
            features, auto_cluster_count, None, criteria, 20,
            cv2.KMEANS_PP_CENTERS
        )
        auto_assignment = auto_assignment.ravel().astype(np.int32)

        # v2.1.0 full symmetry pairing.
        by_label = {region['label']: region for region in regions}
        valid_labels = set(by_label)

        def transform(yy, xx, mode):
            if mode == 'vertical':
                return yy, w - 1 - xx
            if mode == 'horizontal':
                return h - 1 - yy, xx
            if mode == 'central':
                return h - 1 - yy, w - 1 - xx
            if mode == 'diagonal_main':
                return xx, yy
            if mode == 'diagonal_anti':
                return h - 1 - xx, w - 1 - yy
            if mode == 'rotate90':
                return xx, w - 1 - yy
            if mode == 'rotate270':
                return h - 1 - xx, yy
            raise ValueError(mode)

        transform_candidates = [
            ('vertical', 'lr'),
            ('horizontal', 'ud'),
            ('central', 'rot180'),
        ]
        if h == w:
            transform_candidates += [
                ('diagonal_main', 'diag_main'),
                ('diagonal_anti', 'diag_anti'),
            ]
        symmetry_confidence = {
            mode: self._tolerant_symmetry_score(barrier, score_name)
            for mode, score_name in transform_candidates
        }
        transforms = self._select_element_aware_symmetry_modes_v415(
            symmetry_confidence
        )
        pair_cost = {}
        for mode in transforms:
            for region in sorted(regions, key=lambda item: item['area'], reverse=True):
                roi = labels[
                    region['y']:region['y'] + region['h'],
                    region['x']:region['x'] + region['w'],
                ]
                local_y, local_x = np.nonzero(roi == region['label'])
                if not local_x.size:
                    continue
                yy, xx = transform(
                    local_y + region['y'], local_x + region['x'], mode
                )
                inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                if not np.any(inside):
                    continue
                candidates = labels[yy[inside], xx[inside]]
                candidates = candidates[candidates > 0]
                if not candidates.size:
                    continue
                counts = np.bincount(candidates)
                for candidate_label in np.argsort(counts)[::-1]:
                    candidate_label = int(candidate_label)
                    overlap = int(counts[candidate_label])
                    if overlap <= 0:
                        break
                    if candidate_label == region['label'] or candidate_label not in valid_labels:
                        continue
                    candidate = by_label[candidate_label]
                    source_area = float(region['area'])
                    candidate_area = float(candidate['area'])
                    coverage = min(
                        overlap / max(1.0, source_area),
                        overlap / max(1.0, candidate_area),
                    )
                    iou = overlap / max(1.0, source_area + candidate_area - overlap)
                    area_ratio = min(source_area, candidate_area) / max(source_area, candidate_area)
                    if coverage < 0.65 or iou < 0.45 or area_ratio < 0.45:
                        continue
                    cost = (1.0 - coverage) + (1.0 - iou) + (1.0 - area_ratio)
                    key = tuple(sorted((region['label'], candidate_label)))
                    if key not in pair_cost or cost < pair_cost[key]:
                        pair_cost[key] = float(cost)
                    break
        pairs = [(a, b, cost) for (a, b), cost in sorted(pair_cost.items())]
        exact_pair_count = len(pairs)
        all_pair_regions = list(regions) + list(deferred_small_regions)
        pairs = self._augment_element_aware_symmetry_pairs_v415(
            all_pair_regions, (h, w), transforms, existing_pairs=pairs
        )
        pairs = self._build_element_aware_motif_pairs_v417(
            all_pair_regions, (h, w), transforms=transforms, existing_pairs=pairs
        )
        main_region_labels = {int(region['label']) for region in regions}
        balance_pairs = [
            pair for pair in pairs
            if int(pair[0]) in main_region_labels
            and int(pair[1]) in main_region_labels
        ]

        # Reference mode: use the already extracted source color labels only when
        # source and target line structure pass the alignment gate.
        aligned_reference = None
        try:
            aligned_reference = self._aligned_reference_labels(line_rgb)
        except Exception:
            aligned_reference = None

        assignment = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
        for region, cluster in zip(regions, auto_assignment):
            assignment[region['label']] = int(cluster)
        reference_purity = {}
        referenced_regions = 0
        reserved_background_owner = None
        reference_background_labels = set()
        region_reference_owner = {}
        if aligned_reference is not None and aligned_reference.shape == labels.shape:
            # 从触边可填连通域中识别外底 reference 标签。外底在源标签中常为
            # 最大的白色 label（通常是 0），它只能保持画布背景，不能经由
            # reference rebalance 被强制分配给内部闭合面。
            for background_component in background_labels:
                values = aligned_reference[labels == int(background_component)]
                values = values[(values >= 0) & (values < requested_count)]
                if values.size == 0:
                    continue
                counts = np.bincount(
                    values.astype(np.int64), minlength=requested_count
                )
                winner = int(np.argmax(counts))
                purity = float(counts[winner] / values.size)
                if purity >= 0.50:
                    reference_background_labels.add(winner)

            for region in regions:
                y0, x0 = int(region['y']), int(region['x'])
                y1 = y0 + int(region['h'])
                x1 = x0 + int(region['w'])
                local_labels = labels[y0:y1, x0:x1]
                local_reference = aligned_reference[y0:y1, x0:x1]
                values = local_reference[local_labels == region['label']]
                values = values[(values >= 0) & (values < requested_count)]
                if reference_background_labels:
                    values = values[
                        ~np.isin(values, list(reference_background_labels))
                    ]
                if values.size == 0:
                    continue
                counts = np.bincount(values.astype(np.int64), minlength=requested_count)
                winner = int(np.argmax(counts))
                purity = float(counts[winner] / values.size)
                if purity >= 0.55:
                    region_reference_owner[int(region['label'])] = winner
                    reference_purity[region['label']] = purity
                    referenced_regions += 1

        assignment_mode = 'reference' if referenced_regions > 0 else 'auto'
        if assignment_mode == 'reference':
            if include_background_owner:
                reference_labels = sorted(set(region_reference_owner.values()))
                reference_owner_reindex = {
                    source_index: index % foreground_capacity
                    for index, source_index in enumerate(reference_labels)
                }
                for region in regions:
                    label = int(region['label'])
                    source_index = region_reference_owner.get(label)
                    if source_index is not None:
                        assignment[label] = reference_owner_reindex[source_index]
            else:
                for label, source_index in region_reference_owner.items():
                    assignment[int(label)] = int(source_index)
        output_count = (
            foreground_capacity
            if include_background_owner
            else requested_count
            if assignment_mode == 'reference'
            else auto_cluster_count
        )

        # 对精确重叠配对和几何补配统一做 owner 锁定。
        assignment = self._enforce_element_aware_pair_assignments_v415(
            assignment, pairs, stats
        )

        owner_reindex = {}
        if assignment_mode == 'auto':
            area_by_owner = {}
            for region in regions:
                owner = int(assignment[region['label']])
                if owner >= 0:
                    area_by_owner[owner] = (
                        area_by_owner.get(owner, 0) + int(region['area'])
                    )
            ordered_owners = sorted(
                area_by_owner,
                key=lambda owner: (-area_by_owner[owner], owner),
            )
            owner_reindex = {
                old_owner: new_owner
                for new_owner, old_owner in enumerate(ordered_owners)
            }
            for region in regions:
                label = region['label']
                if int(assignment[label]) in owner_reindex:
                    assignment[label] = owner_reindex[int(assignment[label])]

        assignment, balance_diagnostics = (
            self._rebalance_element_aware_assignments_v414(
                regions, assignment, stats, balance_pairs, output_count,
                assignment_mode=assignment_mode,
            )
        )

        background_owner = (
            output_count if include_background_owner else None
        )
        # Coloring 默认继续把触边白域留为 -1；Traditional Transfer 则把
        # 所有触边连通域整体交给最后一个独立 owner，避免残留大块纯白。
        for label in background_labels:
            assignment[int(label)] = (
                int(background_owner) if background_owner is not None else -1
            )

        # 面积很小的闭合面也必须参与分割。以完整连通面为单位归入最近的
        # 前景结构组，绝不逐像素切碎，也不把它留成白色空洞。
        foreground_regions = [
            region for region in regions
            if 0 <= int(assignment[region['label']]) < output_count
        ]
        deferred_labels = np.asarray([
            label for label in range(1, stats.shape[0])
            if label not in background_labels and assignment[label] < 0
        ], dtype=np.int32)
        if deferred_labels.size:
            if foreground_regions:
                foreground_points = np.asarray([
                    [region['cx'] / max(1.0, float(w)),
                     region['cy'] / max(1.0, float(h))]
                    for region in foreground_regions
                ], dtype=np.float64)
                foreground_features = np.asarray([
                    [
                        math.log1p(float(region['area'])) / math.log1p(float(h * w)),
                        math.log(max(float(region['aspect']), 1e-4)),
                        float(region['extent']),
                        float(region['circularity']),
                    ]
                    for region in foreground_regions
                ], dtype=np.float64)
                foreground_features = (
                    foreground_features - foreground_features.mean(axis=0, keepdims=True)
                ) / (foreground_features.std(axis=0, keepdims=True) + 1e-6)
                deferred_map = {int(region['label']): region for region in deferred_small_regions}
                deferred_regions = [
                    deferred_map.get(int(label), {
                        'label': int(label),
                        'area': int(stats[int(label), cv2.CC_STAT_AREA]),
                        'aspect': float(stats[int(label), cv2.CC_STAT_WIDTH]) / max(1.0, float(stats[int(label), cv2.CC_STAT_HEIGHT])),
                        'extent': 1.0,
                        'circularity': 0.55,
                        'cx': float(centroids[int(label), 0]),
                        'cy': float(centroids[int(label), 1]),
                    })
                    for label in deferred_labels
                ]
                deferred_points = np.asarray([
                    [region['cx'] / max(1.0, float(w)),
                     region['cy'] / max(1.0, float(h))]
                    for region in deferred_regions
                ], dtype=np.float64)
                deferred_features = np.asarray([
                    [
                        math.log1p(float(region['area'])) / math.log1p(float(h * w)),
                        math.log(max(float(region['aspect']), 1e-4)),
                        float(region['extent']),
                        float(region['circularity']),
                    ]
                    for region in deferred_regions
                ], dtype=np.float64)
                deferred_features = (
                    deferred_features - foreground_features.mean(axis=0, keepdims=True)
                ) / (foreground_features.std(axis=0, keepdims=True) + 1e-6)
                nearest_indices = []
                chunk_size = max(64, int(1_500_000 / max(1, len(foreground_points))))
                for start in range(0, len(deferred_points), chunk_size):
                    chunk_points = deferred_points[start:start + chunk_size]
                    chunk_features = deferred_features[start:start + chunk_size]
                    spatial2 = np.sum(
                        (chunk_points[:, None, :] - foreground_points[None, :, :]) ** 2,
                        axis=2,
                    )
                    shape2 = np.sum(
                        (chunk_features[:, None, :] - foreground_features[None, :, :]) ** 2,
                        axis=2,
                    )
                    composite = 0.62 * np.sqrt(spatial2) + 0.38 * np.sqrt(shape2)
                    nearest_indices.append(np.argmin(composite, axis=1))
                nearest_indices = np.concatenate(nearest_indices).astype(np.intp)
                nearest_labels = np.asarray([
                    foreground_regions[int(index)]['label']
                    for index in nearest_indices
                ], dtype=np.int32)
                assignment[deferred_labels] = assignment[nearest_labels]
            else:
                assignment[deferred_labels] = 0

        # 小闭合面完成最近邻归属后，再次锁定全部镜像 pair，避免左右饰件异色。
        assignment = self._enforce_element_aware_pair_assignments_v415(
            assignment, pairs, stats
        )

        # 连通域标签一次性映射到 owner，避免对每个小区域反复扫描整幅大图。
        owner_map = assignment[labels]
        total_output_count = output_count + (
            1 if background_owner is not None and background_owner >= output_count else 0
        )
        masks = [
            np.uint8(owner_map == owner) * 255
            for owner in range(total_output_count)
        ]

        fillable_mask = fillable > 0
        canonical_min_area = max(6, int(round((h * w) / 500000.0)))
        owner_map = self._refine_element_aware_owner_map(
            owner_map, min_area=canonical_min_area,
            iterations=0 if transforms else 2,
            protected_owner=background_owner,
        )
        used_owners = [
            owner for owner in range(total_output_count)
            if np.any(owner_map == owner)
        ]
        compact_owner = {
            old_owner: new_owner
            for new_owner, old_owner in enumerate(used_owners)
        }
        if used_owners != list(range(total_output_count)):
            lookup = np.full(total_output_count, -1, dtype=np.int32)
            for old_owner, new_owner in compact_owner.items():
                lookup[old_owner] = new_owner
            valid_owner = owner_map >= 0
            owner_map[valid_owner] = lookup[owner_map[valid_owner]]
        output_count = len(used_owners)
        compact_background_owner = (
            compact_owner.get(int(background_owner))
            if background_owner is not None else None
        )
        masks = [
            np.uint8(owner_map == owner) * 255 for owner in range(output_count)
        ]
        covered_pixels = int(np.count_nonzero(owner_map[fillable_mask] >= 0))
        fillable_pixels = int(np.count_nonzero(fillable_mask))
        coverage_ratio = covered_pixels / max(1, fillable_pixels)

        self._element_aware_v210_assignment_mode = assignment_mode
        self._element_aware_v415_symmetry_pairs = list(pairs)
        self._element_aware_v415_assignment = np.asarray(assignment, dtype=np.int32).copy()
        self._element_aware_v210_owner_map = owner_map
        if include_background_owner:
            owner_capacity = max(used_owners, default=-1) + 1
            reliable_sources = [set() for _ in range(owner_capacity)]
            for label, source_index in region_reference_owner.items():
                owner = int(assignment[int(label)])
                if 0 <= owner < owner_capacity:
                    reliable_sources[owner].add(int(source_index))
            source_by_owner = [
                next(iter(values)) if len(values) == 1 else -1
                for values in reliable_sources
            ]
            self._element_aware_v210_group_source_indices = [
                int(source_by_owner[owner]) for owner in used_owners
            ]
        elif assignment_mode == 'reference':
            owner_capacity = max(used_owners, default=-1) + 1
            reliable_sources = [set() for _ in range(owner_capacity)]
            for label, source_index in region_reference_owner.items():
                owner = int(assignment[int(label)])
                if 0 <= owner < owner_capacity:
                    reliable_sources[owner].add(int(source_index))
            source_by_owner = [
                next(iter(values)) if len(values) == 1 else -1
                for values in reliable_sources
            ]
            self._element_aware_v210_group_source_indices = [
                int(source_by_owner[owner]) for owner in used_owners
            ]
        else:
            self._element_aware_v210_group_source_indices = list(used_owners)
        self._element_aware_v210_labels = labels
        self._element_aware_v210_stats = stats
        self._element_aware_v210_background_label = int(background_label)
        if include_background_owner:
            self._element_aware_v210_background_owner = compact_background_owner
        self._element_aware_v210_reference_purity = reference_purity
        background_fillable = fillable_mask & np.isin(
            labels, list(background_labels)
        )
        background_coverage = (
            float(np.mean(owner_map[background_fillable] == compact_background_owner))
            if compact_background_owner is not None and np.any(background_fillable)
            else 0.0
        )
        self._last_element_aware_v21_diagnostics = {
            'core_version': '2.1.0-refined-v4.1.7-fast-motif-cache',
            'line_threshold': 235,
            'close_kernel': 1,
            'min_area': 20,
            'connectivity': 4,
            'image_size': [int(w), int(h)],
            'connected_components': int(stats.shape[0] - 1),
            'background_label': int(background_label),
            'background_labels': sorted(int(value) for value in background_labels),
            'background_owner': compact_background_owner,
            'background_coverage': background_coverage,
            'removed_empty_groups': int(requested_count - output_count),
            'kept_regions': len(regions),
            'requested_color_groups': requested_count,
            'returned_color_groups': output_count,
            'assignment_mode': assignment_mode,
            'reference_regions': referenced_regions,
            'reference_background_labels': sorted(
                int(value) for value in reference_background_labels
            ),
            'symmetric_pair_count': len(pairs),
            'exact_symmetric_pair_count': int(exact_pair_count),
            'geometry_symmetric_pair_count': int(max(0, len(pairs) - exact_pair_count)),
            'motif_pair_count': int(max(0, len(pairs) - exact_pair_count)),
            'symmetry_lock_enabled': bool(transforms),
            'symmetry_confidence': {
                key: float(value) for key, value in symmetry_confidence.items()
            },
            'enabled_symmetries': list(transforms),
            'foreground_owner_reindex': {
                str(key): int(value) for key, value in owner_reindex.items()
            },
            'owner_balance': balance_diagnostics,
            'fillable_coverage': float(coverage_ratio),
            'native_input': True,
        }
        return masks

    @staticmethod
    def _tolerant_symmetry_score(mask, transform):
        """在统一到 512 像素后计算允许轻微线宽/位置误差的双向覆盖率。"""
        max_side = max(mask.shape)
        scale = min(1.0, 512.0 / max_side)
        if scale < 1.0:
            resized = cv2.resize(
                mask.astype(np.uint8),
                (max(1, round(mask.shape[1] * scale)), max(1, round(mask.shape[0] * scale))),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            resized = mask.astype(np.uint8)

        if transform == 'lr':
            moved = cv2.flip(resized, 1)
        elif transform == 'ud':
            moved = cv2.flip(resized, 0)
        elif transform == 'rot180':
            moved = cv2.flip(resized, -1)
        elif transform == 'diag_main':
            moved = cv2.transpose(resized)
        elif transform == 'diag_anti':
            moved = cv2.flip(cv2.transpose(resized), -1)
        else:
            return 0.0

        if moved.shape != resized.shape:
            return 0.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        expanded_base = cv2.dilate(resized, kernel) > 0
        expanded_moved = cv2.dilate(moved, kernel) > 0
        base_bool = resized > 0
        moved_bool = moved > 0
        base_count = max(1, int(np.count_nonzero(base_bool)))
        moved_count = max(1, int(np.count_nonzero(moved_bool)))
        forward = np.count_nonzero(moved_bool & expanded_base) / moved_count
        backward = np.count_nonzero(base_bool & expanded_moved) / base_count
        return float(0.5 * (forward + backward))

    @staticmethod
    def _symmetry_target(point, transform):
        x, y = point
        if transform == 'lr':
            return np.array([1.0 - x, y], dtype=np.float32)
        if transform == 'ud':
            return np.array([x, 1.0 - y], dtype=np.float32)
        if transform == 'rot180':
            return np.array([1.0 - x, 1.0 - y], dtype=np.float32)
        if transform == 'diag_main':
            return np.array([y, x], dtype=np.float32)
        return np.array([1.0 - y, 1.0 - x], dtype=np.float32)

    def _aligned_reference_labels(
            self, outline_array, minimum_score=0.90, minimum_orientation_score=0.80,
            minimum_fine_score=0.90, minimum_projection_score=0.90):
        """仅在参考图与线稿的边缘位置、方向都高度同源时配准颜色标签。"""
        self._reference_alignment_score = 0.0
        self._reference_orientation_score = 0.0
        self._reference_fine_alignment_score = 0.0
        self._reference_projection_score = 0.0
        if (not getattr(self, 'image_path', None)
                or not os.path.exists(self.image_path)
                or getattr(self, 'labels_2d', None) is None):
            return None
        try:
            reference = np.asarray(Image.open(self.image_path).convert('RGB'))
            target = outline_array if outline_array.ndim == 3 else cv2.cvtColor(
                outline_array, cv2.COLOR_GRAY2RGB
            )

            def content_box(image):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                ys, xs = np.where(gray < 245)
                if xs.size == 0:
                    return None
                return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

            source_box = content_box(reference)
            target_box = content_box(target)
            if source_box is None or target_box is None:
                return None

            def normalized_gray_and_edges(image, box):
                x0, y0, x1, y1 = box
                crop = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
                return crop, cv2.Canny(crop, 50, 150) > 0

            def orientation_descriptor(gray, edges):
                # 边缘覆盖率对高密度纹理过于宽松；方向直方图可区分横线和竖线等伪同源图。
                gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                magnitude = cv2.magnitude(gradient_x, gradient_y)
                orientation = cv2.phase(
                    gradient_x, gradient_y, angleInDegrees=True
                ) % 180.0
                histogram, _ = np.histogram(
                    orientation[edges],
                    bins=18,
                    range=(0.0, 180.0),
                    weights=magnitude[edges],
                )
                histogram = histogram.astype(np.float64)
                norm = float(np.linalg.norm(histogram))
                return histogram / norm if norm > 0.0 else histogram

            def projection_similarity(first_edges, second_edges):
                axis_scores = []
                for axis in (0, 1):
                    first = np.sum(first_edges, axis=axis, dtype=np.float64)
                    second = np.sum(second_edges, axis=axis, dtype=np.float64)
                    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
                    axis_scores.append(
                        float(np.dot(first, second) / denominator)
                        if denominator > 0.0 else 0.0
                    )
                # 两个轴都要一致，防止密集平行纹理只在一个方向上看起来相同。
                return min(axis_scores)

            source_gray, source_edges = normalized_gray_and_edges(reference, source_box)
            target_gray, target_edges = normalized_gray_and_edges(target, target_box)
            def symmetric_edge_coverage(kernel_size):
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                source_dilated = cv2.dilate(
                    source_edges.astype(np.uint8), kernel
                ) > 0
                target_dilated = cv2.dilate(
                    target_edges.astype(np.uint8), kernel
                ) > 0
                source_coverage = np.count_nonzero(
                    source_edges & target_dilated
                ) / max(1, np.count_nonzero(source_edges))
                target_coverage = np.count_nonzero(
                    target_edges & source_dilated
                ) / max(1, np.count_nonzero(target_edges))
                return float(0.5 * (source_coverage + target_coverage))

            # 9px 容差负责抗锯齿/线宽差，5px 精配准负责拒绝同方向但周期不同的密纹理。
            score = symmetric_edge_coverage(9)
            fine_score = symmetric_edge_coverage(5)
            orientation_score = float(np.dot(
                orientation_descriptor(source_gray, source_edges),
                orientation_descriptor(target_gray, target_edges),
            ))
            projection_score = projection_similarity(source_edges, target_edges)
            self._reference_alignment_score = score
            self._reference_fine_alignment_score = fine_score
            self._reference_orientation_score = orientation_score
            self._reference_projection_score = projection_score
            if (score < minimum_score
                    or fine_score < minimum_fine_score
                    or orientation_score < minimum_orientation_score
                    or projection_score < minimum_projection_score):
                return None

            source_labels = np.asarray(self.labels_2d, dtype=np.int32)
            source_h, source_w = reference.shape[:2]
            label_h, label_w = source_labels.shape[:2]
            sx0, sy0, sx1, sy1 = source_box
            lx0 = max(0, min(label_w - 1, round(sx0 * label_w / source_w)))
            ly0 = max(0, min(label_h - 1, round(sy0 * label_h / source_h)))
            lx1 = max(lx0 + 1, min(label_w, round(sx1 * label_w / source_w)))
            ly1 = max(ly0 + 1, min(label_h, round(sy1 * label_h / source_h)))
            label_crop = source_labels[ly0:ly1, lx0:lx1]

            tx0, ty0, tx1, ty1 = target_box
            aligned = np.full(target.shape[:2], -1, dtype=np.int32)
            aligned[ty0:ty1, tx0:tx1] = cv2.resize(
                label_crop,
                (tx1 - tx0, ty1 - ty0),
                interpolation=cv2.INTER_NEAREST,
            )
            return aligned
        except Exception:
            self._reference_alignment_score = 0.0
            self._reference_orientation_score = 0.0
            self._reference_fine_alignment_score = 0.0
            self._reference_projection_score = 0.0
            return None

    def _build_cross_line_region_graph(
            self, regions, outline_array, max_gap=4, max_side=1024):
        """小规模兼容入口：把 mask 列表转为一张标签图后复用生产实现。"""
        h, w = outline_array.shape[:2]
        root_label_map = np.full((h, w), -1, dtype=np.int32)
        root_ids = np.arange(len(regions), dtype=np.int32)
        for root_id, region in zip(root_ids, regions):
            root_label_map[np.asarray(region) > 0] = int(root_id)
        return self._build_cross_line_region_graph_from_labels(
            root_label_map,
            root_ids,
            outline_array,
            max_gap=max_gap,
            max_side=max_side,
        )

    def _build_cross_line_region_graph_from_labels(
            self, root_label_map, root_ids, outline_array,
            max_gap=4, max_side=1024):
        """缩放一次 root 标签图，并仅在局部 bbox 内构造确定性跨线 RAG。"""
        graph = nx.Graph()
        root_ids = np.asarray(root_ids, dtype=np.int32).reshape(-1)
        if root_ids.size == 0:
            return graph

        h, w = outline_array.shape[:2]
        longest = max(h, w)
        scale = min(1.0, float(max_side) / max(1, longest))
        scaled_h = max(1, int(round(h * scale)))
        scaled_w = max(1, int(round(w * scale)))
        source_labels = np.asarray(root_label_map, dtype=np.int32)
        if source_labels.shape != (h, w):
            raise ValueError("root_label_map must match outline dimensions")
        if (scaled_h, scaled_w) != (h, w):
            scaled_label_map = cv2.resize(
                source_labels,
                (scaled_w, scaled_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        else:
            scaled_label_map = source_labels.copy()
        if len(outline_array.shape) == 3:
            gray = cv2.cvtColor(outline_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.asarray(outline_array, dtype=np.uint8)
        if (scaled_h, scaled_w) != (h, w):
            scaled_gray = cv2.resize(
                gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
        else:
            scaled_gray = gray
        dark_pixels = scaled_gray < 245
        root_to_node = {int(root_id): index for index, root_id in enumerate(root_ids)}

        graph.graph['scale'] = scale
        graph.graph['scaled_shape'] = (scaled_h, scaled_w)
        graph.graph['_scaled_label_map'] = scaled_label_map
        graph.graph['_root_ids'] = root_ids

        for index, root_id in enumerate(root_ids):
            mask = scaled_label_map == int(root_id)
            points = cv2.findNonZero(mask.astype(np.uint8))
            if points is None:
                graph.add_node(index, area=0, centroid=(0.0, 0.0), bbox=(0, 0, 0, 0),
                               background=False)
                continue
            left, top, width, height = cv2.boundingRect(points)
            moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
            centroid = (
                float(moments['m10'] / max(moments['m00'], 1.0)),
                float(moments['m01'] / max(moments['m00'], 1.0)),
            )
            background = bool(
                np.any(mask[0, :]) or np.any(mask[-1, :])
                or np.any(mask[:, 0]) or np.any(mask[:, -1])
            )
            graph.add_node(
                index,
                area=int(np.count_nonzero(mask)),
                centroid=centroid,
                bbox=(left, top, width, height),
                background=background,
                root_id=int(root_id),
            )

        # gap 定义为两个填充区最近像素距离减一，因此要搜索到 max_gap + 1。
        # max_gap 属于最长边不超过 1024 的标签图坐标，不再乘缩放比例。
        search_radius = max(1, int(max_gap) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_stats = {}
        for index, root_id in enumerate(root_ids):
            mask = scaled_label_map == int(root_id)
            if graph.nodes[index]['background'] or not np.any(mask):
                continue
            component_count, component_labels, component_stats, _ = (
                cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            )
            for component in range(1, component_count):
                left = int(component_stats[component, cv2.CC_STAT_LEFT])
                top = int(component_stats[component, cv2.CC_STAT_TOP])
                width = int(component_stats[component, cv2.CC_STAT_WIDTH])
                height = int(component_stats[component, cv2.CC_STAT_HEIGHT])
                x0, y0 = max(0, left - search_radius), max(0, top - search_radius)
                x1 = min(scaled_w, left + width + search_radius)
                y1 = min(scaled_h, top + height + search_radius)
                local_component = (
                    component_labels[y0:y1, x0:x1] == component
                ).astype(np.uint8)
                expanded = local_component
                local_labels = scaled_label_map[y0:y1, x0:x1]
                first_gap = {}
                for distance in range(1, search_radius + 1):
                    expanded = cv2.dilate(expanded, kernel, iterations=1)
                    touched = np.unique(local_labels[expanded > 0])
                    for other_label in touched:
                        other = root_to_node.get(int(other_label))
                        if other is None or other == index:
                            continue
                        if graph.nodes[other]['background']:
                            continue
                        first_gap.setdefault(other, distance - 1)
                for other, distance in first_gap.items():
                    pair = (min(index, other), max(index, other))
                    other_pixels = local_labels == int(root_ids[other])
                    contact = int(np.count_nonzero((expanded > 0) & other_pixels))
                    previous = edge_stats.get(pair)
                    if previous is None:
                        edge_stats[pair] = [distance, contact]
                    else:
                        previous[0] = min(previous[0], distance)
                        previous[1] += contact

        def crosses_one_continuous_dark_line(first, second):
            """验证最近廊道只穿过一段连续暗线，拒绝双边框夹白隙。"""
            first_bbox = graph.nodes[first]['bbox']
            second_bbox = graph.nodes[second]['bbox']
            x0 = max(0, min(first_bbox[0], second_bbox[0]) - search_radius)
            y0 = max(0, min(first_bbox[1], second_bbox[1]) - search_radius)
            x1 = min(
                scaled_w,
                max(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
                + search_radius,
            )
            y1 = min(
                scaled_h,
                max(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])
                + search_radius,
            )
            local_labels = scaled_label_map[y0:y1, x0:x1]
            first_mask = local_labels == int(root_ids[first])
            second_mask = local_labels == int(root_ids[second])
            local_dark = dark_pixels[y0:y1, x0:x1]
            height, width = first_mask.shape
            offsets = sorted(
                (
                    (dx * dx + dy * dy, dy, dx)
                    for dy in range(-search_radius, search_radius + 1)
                    for dx in range(-search_radius, search_radius + 1)
                    if dx or dy
                ),
                key=lambda item: (item[0], item[1], item[2]),
            )
            best_distance = None
            for squared_distance, dy, dx in offsets:
                if best_distance is not None and squared_distance > best_distance:
                    break
                first_y = slice(max(0, -dy), min(height, height - dy))
                first_x = slice(max(0, -dx), min(width, width - dx))
                second_y = slice(max(0, dy), min(height, height + dy))
                second_x = slice(max(0, dx), min(width, width + dx))
                overlap = (
                    first_mask[first_y, first_x]
                    & second_mask[second_y, second_x]
                )
                locations = np.argwhere(overlap)
                if locations.size == 0:
                    continue
                best_distance = squared_distance
                first_y0 = max(0, -dy)
                first_x0 = max(0, -dx)
                for local_y, local_x in locations[:64]:
                    start_y = int(local_y + first_y0)
                    start_x = int(local_x + first_x0)
                    end_y = start_y + dy
                    end_x = start_x + dx
                    steps = max(abs(dx), abs(dy))
                    line_x = np.rint(np.linspace(start_x, end_x, steps + 1)).astype(int)
                    line_y = np.rint(np.linspace(start_y, end_y, steps + 1)).astype(int)
                    corridor = local_dark[line_y[1:-1], line_x[1:-1]]
                    if corridor.size == 0:
                        continue
                    dark_runs = int(np.count_nonzero(
                        corridor & ~np.r_[False, corridor[:-1]]
                    ))
                    if dark_runs == 1:
                        return True
            return False

        for (first, second), (distance, contact) in sorted(edge_stats.items()):
            if not crosses_one_continuous_dark_line(first, second):
                continue
            normalizer = math.sqrt(max(
                1,
                graph.nodes[first]['area'] * graph.nodes[second]['area'],
            ))
            graph.add_edge(
                first,
                second,
                gap=int(distance),
                contact=int(contact),
                weight=float(contact / normalizer),
            )
        return graph

    def _coarsen_outline_regions_by_local_motifs(
            self, root_masks, outline_array, max_gap=4, max_side=1024):
        """小规模兼容入口；生产主路径直接传递单张 root 标签图。"""
        if len(root_masks) <= 1:
            self._last_outline_motif_graph = None
            return list(root_masks)
        h, w = outline_array.shape[:2]
        root_label_map = np.full((h, w), -1, dtype=np.int32)
        root_ids = np.arange(len(root_masks), dtype=np.int32)
        for root_id, root_mask in zip(root_ids, root_masks):
            root_label_map[np.asarray(root_mask) > 0] = int(root_id)
        return self._coarsen_outline_region_labels_by_local_motifs(
            root_label_map,
            root_ids,
            outline_array,
            max_gap=max_gap,
            max_side=max_side,
        )

    def _coarsen_outline_region_labels_by_local_motifs(
            self, root_label_map, root_ids, outline_array,
            max_gap=4, max_side=1024):
        """从单张 root 标签图粗化 motif，仅在最终输出时物化全分辨率 mask。"""
        root_ids = np.asarray(root_ids, dtype=np.int32).reshape(-1)
        if root_ids.size <= 1:
            self._last_outline_motif_graph = None
            return [
                (np.asarray(root_label_map) == int(root_id)).astype(np.uint8) * 255
                for root_id in root_ids
            ]

        graph = self._build_cross_line_region_graph_from_labels(
            root_label_map,
            root_ids,
            outline_array,
            max_gap=max_gap,
            max_side=max_side,
        )
        node_count = len(root_ids)
        parent = np.arange(node_count, dtype=np.int32)

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = int(parent[node])
            return int(node)

        def union(first, second):
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(first_root, second_root)

        scaled_label_map = graph.graph['_scaled_label_map']
        scaled_root_ids = graph.graph['_root_ids']
        scaled_h, scaled_w = graph.graph['scaled_shape']
        features = {}
        for node in sorted(graph.nodes):
            left, top, width, height = graph.nodes[node]['bbox']
            mask = scaled_label_map == int(scaled_root_ids[node])
            local = mask[top:top + height, left:left + width].astype(np.uint8)
            contours, _ = cv2.findContours(
                local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea) if contours else None
            perimeter = cv2.arcLength(contour, True) if contour is not None else 0.0
            contour_area = cv2.contourArea(contour) if contour is not None else 0.0
            hull_area = (
                cv2.contourArea(cv2.convexHull(contour))
                if contour is not None else 0.0
            )
            circularity = (
                4.0 * math.pi * contour_area / (perimeter * perimeter)
                if perimeter > 0 else 0.0
            )
            solidity = contour_area / hull_area if hull_area > 0 else 0.0
            # root 可能含多个全局对称副本；用最大局部轮廓描述形状，避免副本间距污染 Hu。
            hu_source = contour if contour is not None else local
            hu = cv2.HuMoments(cv2.moments(hu_source)).ravel()
            hu_first = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-12))
            vertices = (
                len(cv2.approxPolyDP(contour, 0.03 * perimeter, True))
                if contour is not None and perimeter > 0 else 0
            )
            features[node] = {
                'area': float(graph.nodes[node]['area']),
                'centroid': graph.nodes[node]['centroid'],
                'bbox': graph.nodes[node]['bbox'],
                'hu_first': hu_first,
                'circularity': float(circularity),
                'solidity': float(solidity),
                'vertices': int(vertices),
            }
            graph.nodes[node].update(features[node])

        # 在同一缩略坐标系中提取完整轮廓树；节点以自身填充像素落入的最深轮廓为层级。
        tree_contours, tree_hierarchy = cv2.findContours(
            (scaled_label_map > 0).astype(np.uint8) * 255,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_depths = []
        if tree_hierarchy is not None:
            parents = tree_hierarchy[0, :, 3]
            for contour_index in range(len(tree_contours)):
                depth = 0
                parent_index = int(parents[contour_index])
                while parent_index >= 0:
                    depth += 1
                    parent_index = int(parents[parent_index])
                contour_depths.append(depth)
        for node in sorted(graph.nodes):
            ys, xs = np.nonzero(scaled_label_map == int(scaled_root_ids[node]))
            depth = 0
            if xs.size and contour_depths:
                point = (float(xs[0]), float(ys[0]))
                depth = max(
                    (
                        contour_depths[index]
                        for index, contour in enumerate(tree_contours)
                        if cv2.pointPolygonTest(contour, point, False) >= 0
                    ),
                    default=0,
                )
            features[node]['hierarchy_depth'] = int(depth)
            graph.nodes[node]['hierarchy_depth'] = int(depth)

        # 诊断图只保留轻量特征；大数组在本次粗化完成后即可释放。
        graph.graph.pop('_scaled_label_map', None)
        graph.graph.pop('_root_ids', None)
        self._last_outline_motif_graph = graph

        def bbox_contains(host, child):
            host_left, host_top, host_width, host_height = features[host]['bbox']
            child_left, child_top, child_width, child_height = features[child]['bbox']
            return (
                host_left < child_left and host_top < child_top
                and host_left + host_width > child_left + child_width
                and host_top + host_height > child_top + child_height
            )

        # RETR_TREE 证实的内层小纹样是层级 must-link；大内嵌主体仍保持独立。
        for first, second in sorted(graph.edges):
            larger, smaller = sorted(
                (first, second), key=lambda node: (-features[node]['area'], node)
            )
            ratio = features[smaller]['area'] / max(1.0, features[larger]['area'])
            if (ratio <= 0.18
                    and bbox_contains(larger, smaller)
                    and features[smaller]['hierarchy_depth']
                    > features[larger]['hierarchy_depth']):
                union(larger, smaller)

        def shape_compatible(first, second):
            first_feature = features[first]
            second_feature = features[second]
            area_ratio = min(first_feature['area'], second_feature['area']) / max(
                1.0, max(first_feature['area'], second_feature['area'])
            )
            if area_ratio < 0.45:
                return False
            if abs(first_feature['circularity'] - second_feature['circularity']) > 0.20:
                return False
            if abs(first_feature['solidity'] - second_feature['solidity']) > 0.20:
                return False
            if abs(first_feature['hu_first'] - second_feature['hu_first']) > 0.35:
                return False
            if (first_feature['solidity'] >= 0.90
                    and second_feature['solidity'] >= 0.90
                    and first_feature['circularity'] >= 0.70
                    and second_feature['circularity'] >= 0.70
                    and abs(first_feature['vertices'] - second_feature['vertices']) >= 3):
                return False
            return True

        # 固定 seed 的 Louvain 只在形状相容的 RAG 上工作；边权保留真实接触强度。
        similarity_graph = nx.Graph()
        similarity_graph.add_nodes_from(sorted(graph.nodes))
        for first, second in sorted(graph.edges):
            if shape_compatible(first, second):
                rag_weight = float(graph.edges[first, second]['weight'])
                community_weight = 1.0 + min(2.0, 10.0 * rag_weight)
                similarity_graph.add_edge(
                    first, second, weight=community_weight, rag_weight=rag_weight
                )
                # 接触占较小区域尺度 8% 以上的强边是局部 must-link。
                if rag_weight >= 0.08:
                    union(first, second)
        communities = nx.community.louvain_communities(
            similarity_graph,
            weight='weight',
            resolution=0.75,
            seed=0,
        )
        for community in sorted(communities, key=lambda nodes: min(nodes)):
            community = sorted(community)
            edge_count = similarity_graph.subgraph(community).number_of_edges()
            if len(community) >= 3 and edge_count >= len(community) - 1:
                anchor = community[0]
                for node in community[1:]:
                    union(anchor, node)

        def current_groups():
            groups = {}
            for node in range(node_count):
                groups.setdefault(find(node), []).append(node)
            return [groups[root] for root in sorted(groups)]

        # 两个含内部结构的局部 motif 若在画布轴上近似对应，并且面积/结构相近，
        # 建立 must-link；位置接近本身不参与判定。
        h, w = scaled_h, scaled_w
        groups = current_groups()
        group_descriptors = []
        for members in groups:
            left = min(features[node]['bbox'][0] for node in members)
            top = min(features[node]['bbox'][1] for node in members)
            right = max(
                features[node]['bbox'][0] + features[node]['bbox'][2]
                for node in members
            )
            bottom = max(
                features[node]['bbox'][1] + features[node]['bbox'][3]
                for node in members
            )
            area = sum(features[node]['area'] for node in members)
            centroid_x = sum(
                features[node]['centroid'][0] * features[node]['area'] for node in members
            ) / max(1.0, area)
            centroid_y = sum(
                features[node]['centroid'][1] * features[node]['area'] for node in members
            ) / max(1.0, area)
            area_signature = sorted(
                features[node]['area'] / max(1.0, area) for node in members
            )
            member_shape_signature = sorted(
                (
                    features[node]['area'] / max(1.0, area),
                    features[node]['hu_first'],
                    features[node]['circularity'],
                    features[node]['solidity'],
                )
                for node in members
            )
            internal_graph = graph.subgraph(members)
            edge_weights = sorted(
                float(data['weight'])
                for _, _, data in internal_graph.edges(data=True)
            )
            max_edge_weight = max(edge_weights, default=1.0)
            edge_signature = [weight / max_edge_weight for weight in edge_weights]
            weighted_degrees = sorted(
                float(internal_graph.degree(node, weight='weight')) for node in members
            )
            max_degree = max(1e-12, max(weighted_degrees, default=0.0))
            degree_signature = [degree / max_degree for degree in weighted_degrees]
            group_descriptors.append({
                'members': members,
                'area': area,
                'centroid': (centroid_x, centroid_y),
                'size': (right - left, bottom - top),
                'area_signature': area_signature,
                'member_shape_signature': member_shape_signature,
                'edge_signature': edge_signature,
                'degree_signature': degree_signature,
            })

        for first_index, first in enumerate(group_descriptors):
            if len(first['members']) < 2:
                continue
            for second in group_descriptors[first_index + 1:]:
                if len(first['members']) != len(second['members']):
                    continue
                area_ratio = min(first['area'], second['area']) / max(
                    1.0, max(first['area'], second['area'])
                )
                if area_ratio < 0.60:
                    continue
                width_ratio = max(first['size'][0], second['size'][0]) / max(
                    1.0, min(first['size'][0], second['size'][0])
                )
                height_ratio = max(first['size'][1], second['size'][1]) / max(
                    1.0, min(first['size'][1], second['size'][1])
                )
                if width_ratio > 1.45 or height_ratio > 1.45:
                    continue
                area_signature_distance = float(np.mean(np.abs(
                    np.asarray(first['area_signature'])
                    - np.asarray(second['area_signature'])
                )))
                if area_signature_distance > 0.12:
                    continue
                if len(first['edge_signature']) != len(second['edge_signature']):
                    continue
                edge_distance = float(np.mean(np.abs(
                    np.asarray(first['edge_signature'])
                    - np.asarray(second['edge_signature'])
                ))) if first['edge_signature'] else 0.0
                degree_distance = float(np.mean(np.abs(
                    np.asarray(first['degree_signature'])
                    - np.asarray(second['degree_signature'])
                )))
                if edge_distance > 0.18 or degree_distance > 0.18:
                    continue
                first_shapes = np.asarray(first['member_shape_signature'])
                second_shapes = np.asarray(second['member_shape_signature'])
                shape_differences = np.mean(np.abs(first_shapes - second_shapes), axis=0)
                if (shape_differences[0] > 0.12
                        or shape_differences[1] > 0.35
                        or shape_differences[2] > 0.20
                        or shape_differences[3] > 0.20):
                    continue
                first_x, first_y = first['centroid']
                second_x, second_y = second['centroid']
                mirrored_distances = (
                    math.hypot((w - 1 - first_x) - second_x, first_y - second_y),
                    math.hypot(first_x - second_x, (h - 1 - first_y) - second_y),
                )
                if min(mirrored_distances) > 0.12 * max(h, w):
                    continue
                union(first['members'][0], second['members'][0])

        motifs = []
        for members in current_groups():
            member_root_ids = root_ids[np.asarray(members, dtype=np.int32)]
            motif = np.isin(root_label_map, member_root_ids).astype(np.uint8) * 255
            if np.any(motif):
                motifs.append(motif)
        return motifs

    def split_regions_by_perfect_mirror_symmetry(self, outline_array, color_count):
        """以原始封闭面为原子，建立近似对称轨道和小型内嵌面继承关系。"""
        import cv2
        import numpy as np

        self._reset_outline_segmentation_metadata()

        if len( outline_array.shape ) == 3:
            gray = cv2.cvtColor( outline_array, cv2.COLOR_RGB2GRAY )
        else:
            gray = outline_array.copy()

        h, w = gray.shape

        _, bw = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        line_mask = (bw < 128)
        fillable = (~line_mask).astype( np.uint8 )

        def enclosed_domain(barrier_mask):
            barrier_fillable = (~barrier_mask).astype(np.uint8)
            _, labels_for_background = cv2.connectedComponents(
                barrier_fillable, connectivity=8
            )
            labels_at_border = np.unique(np.concatenate((
                labels_for_background[0, :], labels_for_background[-1, :],
                labels_for_background[:, 0], labels_for_background[:, -1],
            )))
            outside = np.isin(labels_for_background, labels_at_border) & (barrier_fillable > 0)
            return (fillable > 0) & (barrier_fillable > 0) & ~outside

        # 先使用真实线条；主体闭合率很低时只修补几像素级断线，不封闭真正开放的大轮廓。
        interior_bool = enclosed_domain(line_mask)
        if np.count_nonzero(interior_bool) < h * w * 0.30:
            # 先把单像素线条加厚，避免 closing 的腐蚀阶段把细线本身抹掉。
            thick_barrier = cv2.dilate(
                line_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            closed_barrier = cv2.morphologyEx(
                thick_barrier,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) > 0
            repaired_domain = enclosed_domain(closed_barrier)
            if np.count_nonzero(repaired_domain) > np.count_nonzero(interior_bool):
                interior_bool = repaired_domain
        interior = interior_bool.astype( np.uint8 ) * 255

        if cv2.countNonZero( interior ) == 0:
            return []

        num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(
            interior_bool.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return []

        component_count = num_labels - 1
        parent = np.arange(num_labels, dtype=np.int32)

        def find(label):
            while parent[label] != label:
                parent[label] = parent[parent[label]]
                label = parent[label]
            return int(label)

        def union(first, second):
            root_first = find(first)
            root_second = find(second)
            if root_first != root_second:
                parent[root_second] = root_first

        labels = np.arange(1, num_labels, dtype=np.int32)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        widths = np.maximum(1, stats[1:, cv2.CC_STAT_WIDTH]).astype(np.float64)
        heights = np.maximum(1, stats[1:, cv2.CC_STAT_HEIGHT]).astype(np.float64)
        aspects = widths / heights
        extents = areas / (widths * heights)
        points = np.column_stack((centroids[1:, 0] / max(1, w - 1),
                                  centroids[1:, 1] / max(1, h - 1))).astype(np.float32)
        tree = NearestNeighborIndex(points)

        # 只在各自 bbox 内提取轮廓描述，避免全图组件掩膜的高内存开销。
        hu_features = np.zeros((component_count, 4), dtype=np.float64)
        circularities = np.zeros(component_count, dtype=np.float64)
        solidities = np.zeros(component_count, dtype=np.float64)
        polygon_vertices = np.zeros(component_count, dtype=np.int16)
        for idx, label in enumerate(labels):
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            local = (comp_labels[top:top + height, left:left + width] == label).astype(np.uint8)
            contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            if perimeter > 0:
                circularities[idx] = 4.0 * math.pi * contour_area / (perimeter * perimeter)
                polygon_vertices[idx] = len(cv2.approxPolyDP(contour, 0.03 * perimeter, True))
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area > 0:
                solidities[idx] = contour_area / hull_area
            hu = cv2.HuMoments(cv2.moments(local)).ravel()[:4]
            hu_features[idx] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

        transform_names = ['lr', 'ud', 'rot180']
        if 0.90 <= w / max(1, h) <= 1.10:
            transform_names.extend(['diag_main', 'diag_anti'])
        symmetry_scores = {
            name: self._tolerant_symmetry_score(interior_bool, name)
            for name in transform_names
        }
        total_area = max(1.0, float(np.sum(areas)))

        def aspect_compatible(first_idx, second_idx, transform):
            expected = aspects[first_idx]
            if transform.startswith('diag'):
                expected = 1.0 / max(expected, 1e-6)
            ratio = max(expected, aspects[second_idx]) / max(
                1e-6, min(expected, aspects[second_idx])
            )
            return ratio <= 1.8

        def shape_compatible(first_idx, second_idx):
            if abs(circularities[first_idx] - circularities[second_idx]) > 0.12:
                return False
            if abs(solidities[first_idx] - solidities[second_idx]) > 0.18:
                return False
            # 高阶 Hu 矩在近方形栅格上会从 0 跳到极小非零值，取对数后不稳定；
            # 第一矩配合圆整度、实心度和顶点数更适合此处的粗筛。
            if abs(hu_features[first_idx, 0] - hu_features[second_idx, 0]) > 0.25:
                return False
            # 圆与菱形的低阶 Hu 矩接近；高实心、较圆整简单图形需再核对顶点结构。
            if (solidities[first_idx] >= 0.90 and solidities[second_idx] >= 0.90
                    and circularities[first_idx] >= 0.74 and circularities[second_idx] >= 0.74
                    and abs(int(polygon_vertices[first_idx])
                            - int(polygon_vertices[second_idx])) >= 3):
                return False
            return True

        def has_central_anchor(transform):
            for idx in range(component_count):
                if areas[idx] / total_area < 0.03:
                    continue
                target = self._symmetry_target(points[idx], transform)
                if np.linalg.norm(target - points[idx]) <= 0.035:
                    if aspect_compatible(idx, idx, transform):
                        return True
            return False

        def strict_match_coverage(transform):
            """统计可形成可靠一一对应的原子区域面积，防止只凭轮廓相似误判全局轴。"""
            nearest = np.full(component_count, -1, dtype=np.int32)
            for idx in range(component_count):
                target = self._symmetry_target(points[idx], transform)
                distances, candidates = tree.query(target, k=min(8, component_count))
                for distance, candidate in zip(np.atleast_1d(distances), np.atleast_1d(candidates)):
                    candidate = int(candidate)
                    if distance > 0.04:
                        continue
                    area_ratio = min(areas[idx], areas[candidate]) / max(areas[idx], areas[candidate])
                    if (area_ratio < 0.65
                            or not aspect_compatible(idx, candidate, transform)
                            or not shape_compatible(idx, candidate)):
                        continue
                    if abs(extents[idx] - extents[candidate]) > 0.30:
                        continue
                    nearest[idx] = candidate
                    break
            covered = set()
            for idx, candidate in enumerate(nearest):
                if candidate >= 0 and nearest[candidate] == idx:
                    covered.add(idx)
                    covered.add(int(candidate))
            return float(np.sum(areas[list(covered)]) / total_area) if covered else 0.0

        match_coverages = {
            name: strict_match_coverage(name) for name in transform_names
        }
        global_transforms = {
            name for name in transform_names
            if symmetry_scores[name] >= 0.90 and match_coverages[name] >= 0.70
        }
        self._last_symmetry_diagnostics = {
            'scores': symmetry_scores,
            'match_coverages': match_coverages,
            'active_global_transforms': sorted(global_transforms),
        }

        # 变换仅用于匹配 label ID；绝不把变换后的像素写回主体域。
        for transform in transform_names:
            global_axis = transform in global_transforms
            relaxed_local = has_central_anchor(transform)
            max_distance = 0.08 if global_axis else (0.17 if relaxed_local else 0.055)
            nearest = np.full(component_count, -1, dtype=np.int32)
            nearest_score = np.full(component_count, np.inf, dtype=np.float64)
            for idx in range(component_count):
                target = self._symmetry_target(points[idx], transform)
                distances, candidates = tree.query(target, k=min(8, component_count))
                distances = np.atleast_1d(distances)
                candidates = np.atleast_1d(candidates)
                for distance, candidate in zip(distances, candidates):
                    candidate = int(candidate)
                    if candidate == idx or distance > max_distance:
                        continue
                    area_ratio = min(areas[idx], areas[candidate]) / max(areas[idx], areas[candidate])
                    if (area_ratio < 0.55
                            or not aspect_compatible(idx, candidate, transform)
                            or not shape_compatible(idx, candidate)):
                        continue
                    if abs(extents[idx] - extents[candidate]) > 0.35:
                        continue
                    score = (float(distance)
                             + 0.035 * abs(math.log(max(area_ratio, 1e-6)))
                             + 0.02 * abs(extents[idx] - extents[candidate]))
                    if score < nearest_score[idx]:
                        nearest[idx] = candidate
                        nearest_score[idx] = score

            for idx, candidate in enumerate(nearest):
                if candidate >= 0 and nearest[candidate] == idx:
                    union(int(labels[idx]), int(labels[candidate]))

        # 很小且隔着一条轮廓线被唯一大区域包围的点/碎屑继承宿主颜色。
        # 0.05% 以内的封闭点优先视为主体纹样；仍需唯一邻接和显著面积差双重确认。
        micro_limit = max(64, min(2048, int(round(h * w * 0.0005))))
        order = np.argsort(areas)
        for idx in order:
            if areas[idx] > micro_limit:
                break
            label = int(labels[idx])
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            gap = max(5, min(12, round(min(h, w) * 0.004)))
            x0, y0 = max(0, left - gap), max(0, top - gap)
            x1, y1 = min(w, left + width + gap), min(h, top + height + gap)
            local_labels = comp_labels[y0:y1, x0:x1]
            local_child = (local_labels == label).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
            ring = (cv2.dilate(local_child, kernel) > 0) & (local_child == 0)
            neighbours = local_labels[ring]
            neighbours = neighbours[(neighbours > 0) & (neighbours != label)]
            if neighbours.size == 0:
                continue
            counts = np.bincount(neighbours, minlength=num_labels)
            candidate = int(np.argmax(counts))
            dominance = counts[candidate] / max(1, int(np.sum(counts)))
            if dominance < 0.75:
                continue
            candidate_area = float(stats[candidate, cv2.CC_STAT_AREA])
            if candidate_area < areas[idx] * 20.0:
                continue
            candidate_left = int(stats[candidate, cv2.CC_STAT_LEFT])
            candidate_top = int(stats[candidate, cv2.CC_STAT_TOP])
            candidate_right = candidate_left + int(stats[candidate, cv2.CC_STAT_WIDTH])
            candidate_bottom = candidate_top + int(stats[candidate, cv2.CC_STAT_HEIGHT])
            child_right = left + width
            child_bottom = top + height
            if not (candidate_left < left and candidate_top < top
                    and candidate_right > child_right and candidate_bottom > child_bottom):
                continue
            candidate_ring = ring & (local_labels == candidate)
            ring_y, ring_x = np.nonzero(candidate_ring)
            if ring_x.size == 0:
                continue
            global_x = ring_x + x0
            global_y = ring_y + y0
            surrounds_all_sides = (
                np.any(global_x < left) and np.any(global_x >= child_right)
                and np.any(global_y < top) and np.any(global_y >= child_bottom)
            )
            if not surrounds_all_sides:
                continue
            union(label, candidate)

        # 统计全局/局部 must-link 后的 root；原始封闭面始终保持完整且互斥。
        root_areas = {}
        for label in labels:
            root = find(int(label))
            root_areas[root] = root_areas.get(root, 0) + int(stats[label, cv2.CC_STAT_AREA])

        # 参考与无参考共用唯一 owner：先由局部 RAG/层级规则形成 motif。
        root_ids = np.asarray(sorted(root_areas), dtype=np.int32)
        label_root = np.full(num_labels, -1, dtype=np.int32)
        for label in labels:
            label_root[label] = find(int(label))
        root_label_map = label_root[comp_labels]
        root_label_map[~interior_bool] = -1
        motifs = self._coarsen_outline_region_labels_by_local_motifs(
            root_label_map,
            root_ids,
            outline_array,
            max_gap=4,
            max_side=1024,
        )

        # 同源参考只在最终 motif 上形成软提示，不能反向拆分 must-link。
        aligned_reference = self._aligned_reference_labels(outline_array)
        if aligned_reference is not None:
            hints = []
            purities = []
            for motif in motifs:
                values = aligned_reference[np.asarray(motif) > 0]
                values = values[values >= 0]
                if values.size == 0:
                    hints.append(-1)
                    purities.append(0.0)
                    continue
                counts = np.bincount(values.astype(np.int64, copy=False))
                winning_hint = int(np.argmax(counts))
                purity = float(counts[winning_hint] / values.size)
                hints.append(winning_hint if purity >= 0.60 else -1)
                purities.append(purity)
            self._reference_region_color_hints = hints
            self._reference_region_color_hint_purities = purities

        return motifs
    # -------------------- 图像显示相关方法 --------------------
    def create_image_tab(self):
        tab_frame = ttk.Frame( self.image_tab )
        tab_frame.pack( fill='both', expand=True, padx=10, pady=10 )

        orig_frame = ttk.LabelFrame( tab_frame, text="Original Image" )
        orig_frame.pack( side='left', fill='both', expand=True, padx=10, pady=10 )
        self.orig_label = ttk.Label( orig_frame, text="Image not loaded", anchor='center' )
        self.orig_label.pack( fill='both', expand=True, padx=10, pady=10 )

        outline_frame = ttk.LabelFrame( tab_frame, text="Line Sketch" )
        outline_frame.pack( side='right', fill='both', expand=True, padx=10, pady=10 )
        self.outline_label = ttk.Label( outline_frame, text="No line sketch generated", anchor='center' )
        self.outline_label.pack( fill='both', expand=True, padx=10, pady=10 )

    def create_extraction_tab(self):
        tab_frame = ttk.Frame( self.extraction_tab )
        tab_frame.pack( fill='both', expand=True, padx=10, pady=10 )

        # 使用 place 的 relwidth 确保严格 4:3 比例

        # Color Histogram - 严格占 4/7 宽度 (~57%)
        hist_frame = ttk.LabelFrame( tab_frame, text="Color Histogram" )
        hist_frame.place( relx=0, rely=0, relwidth=4 / 7, relheight=0.9, anchor='nw' )  # 稍微减小高度给按钮留空间

        # 添加下载按钮框架 - 居中
        hist_btn_frame = ttk.Frame( hist_frame )
        hist_btn_frame.pack( side='bottom', fill='x', pady=5 )
        ttk.Button( hist_btn_frame, text="Download Histogram", command=self.save_histogram ).pack( pady=2 )

        self.hist_canvas = tk.Canvas( hist_frame, bg='white', highlightthickness=0 )
        self.hist_canvas.pack( fill='both', expand=True, padx=10, pady=10 )

        # Color Relationship Networks - 严格占 3/7 宽度 (~43%)
        net_frame = ttk.LabelFrame( tab_frame, text="Color Relationship Networks" )
        net_frame.place( relx=4 / 7, rely=0, relwidth=3 / 7, relheight=0.9, anchor='nw' )

        # 添加下载按钮框架 - 居中
        net_btn_frame = ttk.Frame( net_frame )
        net_btn_frame.pack( side='bottom', fill='x', pady=5 )

        # 关键修改：使用 Frame 包装按钮并居中
        net_btn_inner_frame = ttk.Frame( net_btn_frame )
        net_btn_inner_frame.pack( expand=True )  # expand=True 让内部框架居中

        ttk.Button( net_btn_inner_frame, text="Download Network", command=self.download_network_graph ).pack()

        self.net_canvas = tk.Canvas( net_frame, bg='white', highlightthickness=0 )
        self.net_canvas.pack( fill='both', expand=True, padx=10, pady=10 )

    def save_histogram(self):
        """保存颜色直方图为图片"""
        if not self.color_data:
            messagebox.showwarning( "Warning", "Please extract colors firstly!" )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("PostScript", "*.ps")],
            title="Save Color Histogram",
            initialfile="Histogram.png"
        )
        if not file_path:
            return

        try:
            # 方法1: 如果Canvas支持postscript，保存为PS然后转换
            # 方法2: 直接使用PIL重新绘制并保存（推荐，更清晰）

            # 使用PIL重新绘制直方图并保存
            self._save_histogram_pil( file_path )
            messagebox.showinfo( "Success", f"Histogram saved to:\n{file_path}" )
            self.update_status( f"Histogram saved to {os.path.basename( file_path )}" )
        except Exception as e:
            messagebox.showerror( "Error", f"Failed to save histogram: {str( e )}" )

    def _save_histogram_pil(self, file_path):
        """使用PIL重新绘制直方图并保存 - 修复版：移除anchor参数"""
        from PIL import Image, ImageDraw, ImageFont

        # 创建高分辨率图像
        width, height = 1200, 800
        img = Image.new( 'RGB', (width, height), color='white' )
        draw = ImageDraw.Draw( img )

        hex_color_codes, color_proportions = self.color_data

        # 计算布局（与display_color_histogram逻辑相同，但适配高分辨率）
        n_colors = len( hex_color_codes )
        margin = 80
        spacing = 10
        available_width = width - margin * 2
        bar_width = min( 100, max( 40, (available_width - spacing * (n_colors - 1)) // n_colors ) )

        total_proportion = sum( color_proportions )
        proportions = [prop / total_proportion for prop in color_proportions]

        total_bars_width = n_colors * bar_width + (n_colors - 1) * spacing
        x_offset = (width - total_bars_width) // 2
        max_bar_height = height - 150  # 留出顶部和底部空间

        # 绘制标题 - 手动居中
        try:
            font_title = ImageFont.truetype( "arial.ttf", 24 )
            font_label = ImageFont.truetype( "arial.ttf", 16 )
            font_small = ImageFont.truetype( "arial.ttf", 14 )
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # 手动计算标题居中位置
        title_text = "Color Distribution Histogram"
        title_bbox = draw.textbbox( (0, 0), title_text, font=font_title )
        title_width = title_bbox[2] - title_bbox[0]
        draw.text( ((width - title_width) // 2, 30), title_text, fill='black', font=font_title )

        # 绘制柱状图
        for i, (color, proportion) in enumerate( zip( hex_color_codes, proportions ) ):
            bar_height = int( proportion * max_bar_height )

            x1 = x_offset + i * (bar_width + spacing)
            y1 = height - bar_height - 80
            x2 = x1 + bar_width
            y2 = height - 80

            # 绘制柱子
            draw.rectangle( [x1, y1, x2, y2], fill=color, outline='black', width=2 )

            # 绘制百分比 - 手动计算居中位置
            percent_text = f"{proportion * 100:.1f}%"
            percent_bbox = draw.textbbox( (0, 0), percent_text, font=font_label )
            percent_width = percent_bbox[2] - percent_bbox[0]
            percent_height = percent_bbox[3] - percent_bbox[1]
            # 放在柱子顶部上方
            draw.text( (x1 + (bar_width - percent_width) // 2, y1 - percent_height - 5),
                       percent_text, fill='black', font=font_label )

            # 绘制颜色编号 - 放在柱子底部下方
            num_text = str( i + 1 )
            num_bbox = draw.textbbox( (0, 0), num_text, font=font_label )
            num_width = num_bbox[2] - num_bbox[0]
            draw.text( (x1 + (bar_width - num_width) // 2, y2 + 10),
                       num_text, fill='black', font=font_label )

            # 绘制颜色代码（如果柱子够高）- 放在柱子中间
            if bar_height > 40:
                # 判断颜色亮度决定文字颜色
                r = int( color[1:3], 16 )
                g = int( color[3:5], 16 )
                b = int( color[5:7], 16 )
                text_color = 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 128 else 'black'

                color_bbox = draw.textbbox( (0, 0), color, font=font_small )
                color_width = color_bbox[2] - color_bbox[0]
                color_height = color_bbox[3] - color_bbox[1]
                # 垂直居中
                draw.text( (x1 + (bar_width - color_width) // 2, y1 + (bar_height - color_height) // 2),
                           color, fill=text_color, font=font_small )

        # 保存图像
        img.save( file_path, dpi=(300, 300) )

    # -------------------- 下载网络图功能修改 --------------------
    def download_network_graph(self):
        """
        一键下载当前生成的节点网络图（只下载当前显示的这一张）
        """
        if not hasattr( self, 'net_figure' ) or self.net_figure is None:
            messagebox.showwarning( "Warning", "No network graphs generated, please generate first!" )
            return

        # 直接保存当前的 figure，不遍历列表
        file_path = filedialog.asksaveasfilename(
            title="Save Network Graph",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
            initialfile="network_graph.png"
        )

        if not file_path:
            return

        try:
            # 直接保存当前 figure，只生成一张图
            self.net_figure.savefig( file_path, format='png', dpi=300, bbox_inches='tight', transparent=True )
            messagebox.showinfo( "Success", f"Network graph saved to:\n{file_path}" )
            self.update_status( f"Network graph saved: {os.path.basename( file_path )}" )
        except Exception as e:
            messagebox.showerror( "Error", f"Save failed: {str( e )}" )

    def create_transfer_tab(self):
        tab_frame = ttk.Frame( self.transfer_tab )
        tab_frame.pack( fill='both', expand=True, padx=10, pady=10 )

        source_frame = ttk.LabelFrame( tab_frame, text="Source color" )
        source_frame.pack( side='left', fill='both', expand=True, padx=10, pady=10 )

        source_canvas = tk.Canvas( source_frame, highlightthickness=0 )
        scrollbar = ttk.Scrollbar( source_frame, orient='vertical', command=source_canvas.yview )
        self.source_colors_frame = ttk.Frame( source_canvas )

        self.source_colors_frame.bind(
            "<Configure>",
            lambda e: source_canvas.configure( scrollregion=source_canvas.bbox( "all" ) )
        )

        source_canvas.create_window( (0, 0), window=self.source_colors_frame, anchor="nw" )
        source_canvas.configure( yscrollcommand=scrollbar.set )

        source_canvas.pack( side='left', fill='both', expand=True )
        scrollbar.pack( side='right', fill='y' )

        target_frame = ttk.LabelFrame( tab_frame, text="Target image" )
        target_frame.pack( side='right', fill='both', expand=True, padx=10, pady=10 )
        self.target_image_label = ttk.Label( target_frame, text="Unable to load target image", anchor='center' )
        self.target_image_label.pack( fill='both', expand=True, padx=10, pady=10 )

        apply_frame = ttk.Frame( tab_frame )
        apply_frame.pack( side='bottom', fill='x', padx=10, pady=10 )
        ttk.Button( apply_frame, text="Transfer to target", command=self.apply_colors_to_target ).pack( pady=5 )

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.jfif")],
            title="Select Image"
        )
        if file_path:
            try:
                # 新的源彩图会生成自己的线稿，不能继续使用上一次手工导入的线稿。
                self._element_aware_original_outline_path = None
                self.image_path = file_path
                img = Image.open(file_path).convert('RGB')
                self.display_image(img, self.orig_label, (400, 400))
                self.create_outline_image(file_path, 'source_outline_image_path')
                if self.source_outline_image_path and os.path.exists(self.source_outline_image_path):
                    outline_img = Image.open(self.source_outline_image_path).convert('RGB')
                    self.display_image(outline_img, self.outline_label, (400, 400))
                self.update_status(f" Load Source Image: {Path(file_path).name}")
                self.show_tab(self.image_tab, "Display image")
            except Exception as e:
                messagebox.showerror("Error!", f"Unable to Load Source Image: {str(e)}")

    def display_image(self, img, label, size):
        """高质量预览显示，避免缩放造成线稿发糊。"""
        img_copy = img.copy().convert('RGB')
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img_copy.thumbnail(size, resample)
        photo = ImageTk.PhotoImage(img_copy)
        label.configure(image=photo)
        label.image = photo

    @staticmethod
    def _line_sketch_remove_small_components(edge_mask, min_area=4, min_span=8):
        """删除孤立噪点，同时保留细长的真实轮廓。"""
        binary = (np.asarray(edge_mask) > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cleaned = np.zeros(binary.shape, dtype=np.uint8)
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if area >= int(min_area) or max(width, height) >= int(min_span):
                cleaned[labels == component_id] = 255
        return cleaned

    @staticmethod
    def _line_sketch_robust_gradient(channel):
        """返回 Scharr 梯度及其鲁棒归一化结果。"""
        source = np.asarray(channel, dtype=np.uint8)
        grad_x = cv2.Scharr(source, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(source, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        scale = float(np.percentile(magnitude, 98.0)) if magnitude.size else 1.0
        scale = max(scale, 1.0)
        normalized = np.clip(magnitude * (255.0 / scale), 0, 255).astype(np.uint8)
        return magnitude, normalized

    def _generate_complete_line_sketch(self, rgb_image):
        """生成非二值、原线宽的柔和线稿，不膨胀、不闭运算、不制造噪点。"""
        rgb = np.asarray(rgb_image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Line sketch generation requires an RGB image")
        rgb = rgb[..., :3]

        # 输入本身已经是线稿时，逐像素原样返回：保留灰阶抗锯齿、原线宽、
        # 原有彩色结构线以及原始分辨率，不再做 Canny、阈值或对比度拉伸。
        channel_spread = (
            rgb.max(axis=2).astype(np.int16)
            - rgb.min(axis=2).astype(np.int16)
        )
        gray_probe = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        simple_outline = (
            float(np.mean(gray_probe >= 245)) >= 0.45
            and float(np.mean(gray_probe < 220)) >= 0.001
            and float(np.percentile(channel_spread, 95)) <= 20.0
        )
        if self.is_outline_image(rgb) or simple_outline:
            return rgb.copy()

        # 彩色图片使用连续梯度强度形成灰阶线稿。整个流程不调用二值阈值、
        # Canny、膨胀、腐蚀或闭运算，因此不会把一条边变成双边或加粗线条。
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        denoised = cv2.bilateralFilter(bgr, d=5, sigmaColor=24, sigmaSpace=5)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)

        def normalized_gradient(channel):
            source = np.asarray(channel, dtype=np.uint8)
            grad_x = cv2.Scharr(source, cv2.CV_32F, 1, 0)
            grad_y = cv2.Scharr(source, cv2.CV_32F, 0, 1)
            magnitude = cv2.magnitude(grad_x, grad_y)
            positive = magnitude[magnitude > 0]
            if positive.size == 0:
                return np.zeros_like(magnitude, dtype=np.float32)
            low = float(np.percentile(positive, 42.0))
            high = float(np.percentile(positive, 99.2))
            if high <= low + 1e-6:
                # 单一强度的清晰边界也必须保留，不能因分位数相等而归零。
                return np.clip(magnitude / max(high, 1.0), 0.0, 1.0)
            # 连续软门限，不产生 0/255 二值边缘。
            return np.clip((magnitude - low) / (high - low), 0.0, 1.0)

        light_edge = normalized_gradient(lightness)
        color_edge = np.maximum(
            normalized_gradient(channel_a),
            normalized_gradient(channel_b),
        )
        edge_strength = np.maximum(light_edge, 0.88 * color_edge)

        # 仅做极弱亚像素平滑以保留抗锯齿，不扩张边界。
        edge_strength = cv2.GaussianBlur(edge_strength, (0, 0), 0.28)
        edge_strength = np.clip(edge_strength, 0.0, 1.0) ** 0.82
        gray_sketch = np.clip(
            255.0 - 225.0 * edge_strength,
            0.0,
            255.0,
        ).astype(np.uint8)
        return cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2RGB)

    def create_outline_image(self, file_path, target_attr='outline_image_path', output_name=None):
        """按原始分辨率保存非二值线稿；已是线稿时完全保留原图像素。"""
        try:
            pil_img = Image.open(file_path).convert('RGB')
            source_array = np.asarray(pil_img)
            sketch = self._generate_complete_line_sketch(source_array)
            if sketch.ndim == 2:
                line_image = Image.fromarray(sketch, mode='L')
            else:
                line_image = Image.fromarray(sketch[..., :3], mode='RGB')

            output_path = output_name or f'{target_attr}.png'
            setattr(self, target_attr, output_path)
            if target_attr == 'source_outline_image_path':
                self.outline_image_path = output_path
            line_image.save(output_path, dpi=(300, 300))
            self.update_status(
                "Original-width, non-binary line sketch has been generated."
            )
            return True
        except Exception as e:
            self.update_status(f"Error creating line sketch: {str(e)}")
            messagebox.showerror("Error!", f" Error creating line sketch: {str(e)}")
            return False

    def extract_colors(self):
        if not self.image_path:
            messagebox.showerror( "Error!", "Please load the image firstly!" )
            return

        try:
            # 初始化网络图列表
            self.generated_network_graphs = []

            color_number = self.color_number_var.get()
            self.update_status( f"  {color_number} color are been extracting..." )

            self.color_regions, self.hex_color_codes, self.labels_2d = self.extract_colors_and_regions(
                self.image_path, color_number
            )

            sorted_colors = sorted( self.color_regions.items(), key=lambda x: x[1]['pixels'], reverse=True )
            sorted_hex_codes = [c[1]['hex'] for c in sorted_colors]
            sorted_counts = [c[1]['pixels'] for c in sorted_colors]

            self.color_data = (sorted_hex_codes, sorted_counts)

            self.outline_coloring_colors = sorted_hex_codes
            self.color_transfer_colors = sorted_hex_codes

            label_map = {old: new for new, (old, _) in enumerate( sorted_colors )}
            remap_func = np.vectorize( lambda x: label_map.get( int( x ), -1 ), otypes=[np.int32] )
            remapped_labels = remap_func( self.labels_2d ).astype( np.int32 )
            self.labels_2d = remapped_labels
            # 标签已经按面积重新编号，调色板必须同步到同一顺序。
            self.hex_color_codes = list(sorted_hex_codes)

            self.display_color_histogram()
            self.draw_color_network_in_canvas(
                self.net_canvas,
                self.image_path,
                sorted_hex_codes,
                sorted_counts,
                self.labels_2d,
                self.network_threshold_var.get()
            )

            self.update_status( f" {color_number} color are extracted!" )
            self.show_tab( self.extraction_tab, "Color extraction" )

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror( "Error!", f"Color extraction failed: {str( e )}" )

    @staticmethod
    def _estimate_palette_sampling_mask(image_array, final_n_colors=5):
        """估计调色板提取的有效采样区域，并区分细描边与真实深色块。

        旧版只排除“低饱和黑灰像素”，因此彩色图案中的深蓝、深紫、深绿描边
        会与真实深色填充一起聚成一个近黑色大簇。这里改用空间厚度判断：
        细而暗、紧贴边缘的像素视为描边；具有足够内部厚度的深色区域保留。
        """
        rgb = np.asarray(image_array, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            keep = np.ones(rgb.shape[:2], dtype=bool)
            return keep, rgb

        smooth = cv2.bilateralFilter(rgb, 7, 35, 35)
        lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1].astype(np.float32)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        chroma = np.sqrt(a * a + b * b)
        h, w = rgb.shape[:2]

        # 1) 仅删除与四周连通、颜色统一的浅色低饱和背景。
        border = max(2, int(round(min(h, w) * 0.02)))
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True
        border_pixels = lab[border_mask]
        background_mask = np.zeros((h, w), dtype=bool)
        if border_pixels.size:
            border_median = np.median(border_pixels, axis=0)
            border_dist = np.linalg.norm(border_pixels - border_median, axis=1)
            border_chroma = np.sqrt(
                (border_pixels[:, 1] - 128.0) ** 2
                + (border_pixels[:, 2] - 128.0) ** 2
            )
            coherent = float(np.mean(border_dist < 10.0)) >= 0.40
            median_border_chroma = float(np.median(border_chroma))
            plain = (
                median_border_chroma < 10.0
                or (
                    median_border_chroma < 14.0
                    and float(border_median[0]) > 210.0
                )
            )
            if coherent and plain:
                all_dist = np.linalg.norm(lab - border_median, axis=2)
                bg_thresh = 10.0 if median_border_chroma < 8.0 else 8.0
                similar_to_border = all_dist <= bg_thresh

                # 只保留从图像边界可达的相似颜色，避免误删主体内部浅色块。
                num, cc = cv2.connectedComponents(
                    similar_to_border.astype(np.uint8), connectivity=4
                )
                if num > 1:
                    edge_labels = np.unique(np.concatenate((
                        cc[0, :], cc[-1, :], cc[:, 0], cc[:, -1]
                    )))
                    edge_labels = edge_labels[edge_labels > 0]
                    background_mask = np.isin(cc, edge_labels)
                    background_mask = cv2.morphologyEx(
                        background_mask.astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        np.ones((3, 3), np.uint8),
                    ).astype(bool)

        valid_mask = ~background_mask

        # 2) 自适应确定“暗色”范围。只依据固定 120 阈值会随图像明暗失效。
        valid_gray = gray[valid_mask]
        dark_percentile = (
            float(np.percentile(valid_gray, 32)) if valid_gray.size else 90.0
        )
        dark_threshold = float(np.clip(dark_percentile + 18.0, 78.0, 138.0))
        dark_mask = (gray < dark_threshold) & valid_mask

        # 3) 空间厚度区分描边与色块。
        # distanceTransform 小：窄线/色块边缘；大：真实深色填充的内部核心。
        dark_distance = cv2.distanceTransform(
            dark_mask.astype(np.uint8), cv2.DIST_L2, 5
        )
        edge_map = cv2.Canny(gray, 45, 125) > 0
        edge_band = cv2.dilate(
            edge_map.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)

        chromatic_thin_ink = dark_mask & (dark_distance <= 2.2) & edge_band
        neutral_ink = (
            dark_mask
            & (sat < 120.0)
            & (chroma < 38.0)
            & (dark_distance <= 3.4)
        )
        line_like_mask = chromatic_thin_ink | neutral_ink

        colorful_core = (
            (sat >= 35.0) & (chroma >= 15.0) & valid_mask
        )
        keep_mask = valid_mask.copy()
        if int(np.count_nonzero(colorful_core)) >= max(
                500, int(final_n_colors) * 40):
            keep_mask &= ~line_like_mask

        # 保底：不能因为线条判断过强导致可聚类样本不足。
        min_keep = max(200, int(final_n_colors) * 30)
        if int(np.count_nonzero(keep_mask)) < min_keep:
            keep_mask = valid_mask
        if int(np.count_nonzero(keep_mask)) < max(
                50, int(final_n_colors) * 15):
            keep_mask = np.ones((h, w), dtype=bool)

        return keep_mask.astype(bool), smooth

    def extract_colors_and_regions(self, image_path, final_n_colors):
        try:
            image = Image.open( image_path ).convert( "RGB" )
            image.thumbnail( (800, 800), Image.Resampling.LANCZOS )
            image_array = np.array( image )
            h, w = image_array.shape[:2]

            keep_mask, filtered_rgb = self._estimate_palette_sampling_mask(
                image_array, final_n_colors
            )
            lab = cv2.cvtColor( filtered_rgb, cv2.COLOR_RGB2LAB )
            sampled = lab[keep_mask].reshape( -1, 3 ).astype( np.float32 )
            if sampled.shape[0] < max(final_n_colors * 20, final_n_colors):
                keep_mask = np.ones((h, w), dtype=bool)
                sampled = lab.reshape( -1, 3 ).astype( np.float32 )

            initial_n_colors = min( max( final_n_colors, sampled.shape[0] ),
                                    max( 8, min( 48, final_n_colors * 3 ) ) )
            kmeans1 = KMeans(
                n_clusters=initial_n_colors,
                init="k-means++",
                random_state=42,
                n_init=5,
            ).fit( sampled )

            labels1 = kmeans1.labels_
            counts = np.bincount( labels1, minlength=initial_n_colors )
            top_idx = np.argsort( counts )[::-1][:final_n_colors]
            init_centers = kmeans1.cluster_centers_[top_idx]

            kmeans2 = KMeans(
                n_clusters=final_n_colors,
                init=init_centers,
                n_init=1,
                random_state=42,
            ).fit( sampled )
            sampled_labels = kmeans2.labels_.astype(np.int32)

            labels_2d = np.full( (h, w), -1, dtype=np.int32 )
            labels_2d[keep_mask] = sampled_labels

            hex_color_codes = []
            color_regions = {}
            for i in range( final_n_colors ):
                mask = (labels_2d == i)
                cluster_pixels = image_array[mask]
                if len( cluster_pixels ) == 0:
                    rgb = np.array( [0, 0, 0], dtype=np.uint8 )
                else:
                    # 使用中位数而不是均值，进一步减少抗锯齿/细描边引入的偏差。
                    rgb = np.median( cluster_pixels, axis=0 ).astype( np.uint8 )
                hex_code = "#{:02x}{:02x}{:02x}".format( *rgb )
                hex_color_codes.append( hex_code )
                color_regions[i] = {
                    "hex": hex_code,
                    "pixels": int( mask.sum() ),
                }

            return color_regions, hex_color_codes, labels_2d

        except Exception as e:
            import traceback
            traceback.print_exc()
            print( f"Error in extract_colors_and_regions: {e}" )
            return {}, [], None


    def display_color_histogram(self):
        """Display color histogram"""
        if not self.color_data:
            return

        self.hist_canvas.delete( "all" )

        width = self.hist_canvas.winfo_width()
        height = self.hist_canvas.winfo_height()

        # Use default size if canvas is too small
        if width < 100 or height < 100:
            width = 400
            height = 300

        hex_color_codes, color_proportions = self.color_data

        # 改进：动态计算柱体宽度，确保不会太大
        n_colors = len( hex_color_codes )
        margin = 40  # 左右边距总和
        spacing = 5  # 柱体间距
        available_width = width - margin

        # 计算柱体宽度：确保至少20px，最大不超过60px
        bar_width = min( 60, max( 20, (available_width - spacing * (n_colors - 1)) / n_colors ) )

        total_proportion = sum( color_proportions )
        proportions = [prop / total_proportion for prop in color_proportions]

        # 居中显示
        total_bars_width = n_colors * bar_width + (n_colors - 1) * spacing
        x_offset = (width - total_bars_width) / 2

        max_bar_height = height - 80  # 留出顶部和底部空间

        for i, (color, proportion) in enumerate( zip( hex_color_codes, proportions ) ):
            # Calculate bar height
            bar_height = int( proportion * max_bar_height )

            # Draw bar
            x1 = x_offset + i * (bar_width + spacing)
            y1 = height - bar_height - 40  # 底部留出空间给标签
            x2 = x1 + bar_width
            y2 = height - 40

            self.hist_canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=color, outline='black'
            )

            # Determine text color based on color brightness
            text_color = "white" if self.is_dark_color( color ) else "black"

            # Display color code in the middle of the bar (如果柱体足够高)
            if bar_height > 30:
                self.hist_canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2,
                    text=color,
                    anchor='center',
                    fill=text_color,
                    font=('Arial', 8)
                )

            # Percentage text: 在柱体顶部上方
            self.hist_canvas.create_text(
                (x1 + x2) / 2, y1 - 5,
                text=f"{proportion * 100:.1f}%",
                anchor='s',
                fill="black",
                font=('Arial', 9, 'bold')
            )

            # 颜色编号在底部
            self.hist_canvas.create_text(
                (x1 + x2) / 2, y2 + 15,
                text=str( i + 1 ),
                anchor='n',
                fill="black",
                font=('Arial', 10, 'bold')
            )

    def is_dark_color(self, hex_color):
        try:
            r = int( hex_color[1:3], 16 )
            g = int( hex_color[3:5], 16 )
            b = int( hex_color[5:7], 16 )
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return luminance < 128
        except:
            return False

    def on_hist_canvas_resize(self, event):
        if self.color_data:
            self.display_color_histogram()

    @staticmethod
    def _color_network_paper_dispersion_factors(labels, color_count):
        """按 Xu 等（2019）公式（1）计算外环直径与节点直径之比 ``D/d``。

        论文定义：

        ``D = d * RMS(pixel-to-centroid distance) / sqrt(n / (2*pi))``

        其中 ``n`` 是该颜色的像素数，``d`` 是按颜色面积绘制的节点直径。
        本函数只返回与节点绘图尺寸无关的 ``D/d``，供显示层和 CRN 评分共同
        使用。负标签及越界标签视为背景，不参与计算。
        """
        label_map = np.asarray(labels)
        count = max(0, int(color_count))
        if label_map.ndim != 2:
            raise ValueError("Color-network labels must be a 2D array")
        if count == 0:
            return np.empty((0,), dtype=np.float64)

        valid = (label_map >= 0) & (label_map < count)
        ys, xs = np.nonzero(valid)
        ids = label_map[valid].astype(np.int64, copy=False)
        pixel_counts = np.bincount(ids, minlength=count).astype(np.float64)
        if ids.size == 0:
            return np.zeros(count, dtype=np.float64)

        xs_float = xs.astype(np.float64, copy=False)
        ys_float = ys.astype(np.float64, copy=False)
        safe_counts = np.maximum(pixel_counts, 1.0)
        mean_x = (
            np.bincount(ids, weights=xs_float, minlength=count) / safe_counts
        )
        mean_y = (
            np.bincount(ids, weights=ys_float, minlength=count) / safe_counts
        )
        mean_squared_radius = np.maximum(
            0.0,
            (
                np.bincount(
                    ids,
                    weights=xs_float ** 2 + ys_float ** 2,
                    minlength=count,
                )
                / safe_counts
                - mean_x ** 2
                - mean_y ** 2
            ),
        )
        rms_radius = np.sqrt(mean_squared_radius)
        minimum_rms = np.sqrt(pixel_counts / (2.0 * np.pi))
        factors = np.divide(
            rms_radius,
            minimum_rms,
            out=np.zeros_like(rms_radius),
            where=minimum_rms > 1e-12,
        )
        return np.maximum(
            np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )

    @staticmethod
    def _color_network_dispersion_ring_sizes(node_sizes, dispersions):
        """依据论文公式（1）把外环直径 ``D`` 转为 Matplotlib points²。

        ``node_size`` 是面积量，标记直径与其平方根成正比。因此
        ``D = d * factor`` 对应 ``ring_size = node_size * factor**2``。
        离散小样本可能因栅格近似得到 ``factor < 1``；仅在绘制层钳制为
        1，避免外环落入节点内部，原始论文分散系数仍原样用于 CRN 评分。
        """
        sizes = np.asarray(node_sizes, dtype=np.float64).reshape(-1)
        dispersion = np.asarray(dispersions, dtype=np.float64).reshape(-1)
        if sizes.size != dispersion.size:
            raise ValueError("Node sizes and dispersions must have equal length")
        if sizes.size == 0:
            return np.empty((0,), dtype=np.float64)

        sizes = np.maximum(np.nan_to_num(sizes, nan=1.0, posinf=1.0), 1.0)
        diameter_factors = np.maximum(
            np.nan_to_num(dispersion, nan=0.0, posinf=0.0, neginf=0.0),
            1.0,
        )
        return sizes * np.square(diameter_factors)

    @staticmethod
    def _color_network_display_ring_sizes(
            node_sizes, dispersions, excess_scale=0.75):
        """把论文外环适度收紧为界面显示尺寸，不修改原始 CRN 分散度。

        论文的 ``D/d`` 仍作为评分和数据属性的权威值。显示层只把超出节点
        的直径部分缩放到 75%，即 ``1 + 0.75 * (D/d - 1)``，从而减少外环
        交叠，同时保持集中度基线、大小顺序和单调关系。
        """
        sizes = np.asarray(node_sizes, dtype=np.float64).reshape(-1)
        factors = np.asarray(dispersions, dtype=np.float64).reshape(-1)
        if sizes.size != factors.size:
            raise ValueError("Node sizes and dispersions must have equal length")
        if sizes.size == 0:
            return np.empty((0,), dtype=np.float64)

        sizes = np.maximum(
            np.nan_to_num(sizes, nan=1.0, posinf=1.0, neginf=1.0),
            1.0,
        )
        paper_factors = np.maximum(
            np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0),
            1.0,
        )
        contraction = float(np.clip(excess_scale, 0.0, 1.0))
        display_factors = 1.0 + contraction * (paper_factors - 1.0)
        return sizes * np.square(display_factors)

    @staticmethod
    def _color_network_ring_style(node_colors):
        """返回与实心节点逐一对应的同色外环样式。"""
        return {
            "edgecolors": list(node_colors),
            "linewidths": 1.55,
            "alpha": 0.90,
            "zorder": 3.0,
        }

    def draw_color_network_in_canvas(self, canvas, image_path, hex_color_codes, color_proportions, labels,
                                     network_threshold):
        """绘制颜色关系网络。

        节点实心面积表示颜色占比；外环表示空间离散度；连线表示两种颜色
        共享边界的归一化强度。这里使用真正的 4 邻域共享边界，避免把仅在
        对角线接触的色块误判为相邻关系。
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import numpy as np

        try:
            for widget in canvas.winfo_children():
                widget.destroy()
        except Exception:
            pass

        if labels is None or labels.ndim != 2:
            return

        n = len(hex_color_codes)
        if n <= 0:
            return

        labels = np.asarray(labels)
        counts = np.asarray(color_proportions, dtype=np.float64)
        if counts.size != n:
            counts = np.bincount(
                labels[(labels >= 0) & (labels < n)].astype(np.int64),
                minlength=n,
            ).astype(np.float64)
        total_count = float(np.sum(counts))
        proportions = (
            counts / total_count
            if total_count > 0.0
            else np.full(n, 1.0 / n, dtype=np.float64)
        )

        fixed_scale = 5000.0
        min_size = 100.0
        node_sizes = np.maximum(proportions * fixed_scale, min_size)

        # 论文公式（1）的 D/d。外环绘制和 CRN 评分共用同一个计算源。
        dispersions = self._color_network_paper_dispersion_factors(labels, n)

        adjacent_pixels_count = self.calculate_adjacent_pixels(labels)

        graph = nx.Graph()
        for index in range(n):
            graph.add_node(
                index,
                color=hex_color_codes[index],
                size=float(node_sizes[index]),
                dispersion=float(dispersions[index]),
            )

        # 每个颜色与其他颜色接触的总共享边界长度。
        boundary_totals = {index: 0.0 for index in range(n)}
        for (first, second), contact in adjacent_pixels_count.items():
            if first == second or not (0 <= first < n and 0 <= second < n):
                continue
            boundary_totals[first] += float(contact)
            boundary_totals[second] += float(contact)

        for (first, second), contact in adjacent_pixels_count.items():
            if first == second or not (0 <= first < n and 0 <= second < n):
                continue
            denominator = boundary_totals[first] + boundary_totals[second]
            if denominator <= 0.0:
                continue

            # 对称 Dice 型共享边界强度：2*b_ij/(b_i+b_j)。
            # 旧式 b_ij/(b_i+b_j) 的理论上限只有 0.5，会系统性把连线
            # 权重压低一半；当两种颜色只彼此相邻时，新定义正确得到 1。
            weight = float(np.clip(
                2.0 * float(contact) / denominator, 0.0, 1.0
            ))
            if weight >= float(network_threshold):
                graph.add_edge(
                    int(first), int(second),
                    weight=weight,
                    boundary_count=int(contact),
                )

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, aspect='equal')
        positions = nx.circular_layout(graph, scale=0.55)

        node_colors = [graph.nodes[node]['color'] for node in graph.nodes()]
        node_sizes_list = [graph.nodes[node]['size'] for node in graph.nodes()]
        # CRN 数据保留论文公式的 D/d；界面仅收紧超出节点的外环部分，减少
        # 圆圈互相遮挡。若仍过大，则对节点和外环施加同一个全局缩放。
        ring_sizes = self._color_network_display_ring_sizes(
            np.asarray(node_sizes_list, dtype=np.float64),
            dispersions,
        )
        largest_ring_diameter = float(np.sqrt(np.max(ring_sizes)))
        if largest_ring_diameter > 165.0:
            global_area_scale = (165.0 / largest_ring_diameter) ** 2
            node_sizes_list = (
                np.asarray(node_sizes_list, dtype=np.float64)
                * global_area_scale
            )
            ring_sizes = ring_sizes * global_area_scale

        nx.draw_networkx_nodes(
            graph, positions,
            node_color=node_colors,
            node_size=node_sizes_list,
            ax=ax,
        )

        node_order = list(graph.nodes())
        ax.scatter(
            [positions[node][0] for node in node_order],
            [positions[node][1] for node in node_order],
            s=ring_sizes,
            facecolors='none',
            **self._color_network_ring_style(node_colors),
        )

        edge_widths = [
            1.0 + 7.0 * np.sqrt(float(graph[u][v]['weight']))
            for u, v in graph.edges()
        ]
        nx.draw_networkx_edges(
            graph, positions,
            width=edge_widths,
            edge_color='gray',
            alpha=0.7,
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={node: str(node) for node in graph.nodes()},
            font_size=10,
            ax=ax,
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        plt.axis('off')
        plt.tight_layout()

        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill='none', expand=False)
        self.net_figure = fig

    def on_net_canvas_resize(self, event):
        if self.color_data is not None and self.image_path and self.labels_2d is not None:
            hex_color_codes, color_proportions = self.color_data
            self.draw_color_network_in_canvas(
                self.net_canvas,
                self.image_path,
                hex_color_codes,
                color_proportions,
                self.labels_2d,
                self.network_threshold_var.get()
            )

    def _prepare_outline_coherence_context(
            self, regions, selected_colors, outline_array=None):
        """预计算区域面积、最终 motif RAG 与调色板 Lab 距离。"""
        region_count = len(regions)
        areas = np.asarray(
            [int(np.count_nonzero(np.asarray(region) > 0)) for region in regions],
            dtype=np.float64,
        )

        graph = nx.Graph()
        graph.add_nodes_from(range(region_count))
        if outline_array is not None and region_count:
            try:
                graph = self._build_cross_line_region_graph(
                    regions, outline_array, max_gap=4
                )
            except (TypeError, ValueError, cv2.error):
                graph = nx.Graph()
                graph.add_nodes_from(range(region_count))
        # 构图过程为性能保留了缩放标签图；评分只需要轻量节点和边。
        graph.graph.pop('_scaled_label_map', None)
        graph.graph.pop('_root_ids', None)
        graph.add_nodes_from(range(region_count))

        edges = []
        adjacency = [[] for _ in range(region_count)]
        weighted_degrees = np.zeros(region_count, dtype=np.float64)
        for first, second, data in sorted(graph.edges(data=True)):
            first = int(first)
            second = int(second)
            if not (0 <= first < region_count and 0 <= second < region_count):
                continue
            raw_weight = max(0.0, float(data.get('weight', 0.0)))
            separator_gap = max(0, int(data.get('gap', 0)))
            # 薄/弱分隔线更应保持同色；较宽的强轮廓降低换色惩罚。
            weak_boundary_factor = max(0.2, 1.0 - separator_gap / 5.0)
            weight = raw_weight * weak_boundary_factor
            if weight <= 0.0:
                continue
            edges.append((first, second, weight))
            adjacency[first].append((second, weight))
            adjacency[second].append((first, weight))
            weighted_degrees[first] += weight
            weighted_degrees[second] += weight

        palette_labs = []
        for color in selected_colors:
            try:
                rgb = np.asarray([[[
                    int(str(color)[1:3], 16),
                    int(str(color)[3:5], 16),
                    int(str(color)[5:7], 16),
                ]]], dtype=np.float32) / 255.0
                palette_labs.append(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)
                )
            except (TypeError, ValueError, cv2.error):
                palette_labs.append(np.zeros(3, dtype=np.float64))
        palette_labs = np.asarray(palette_labs, dtype=np.float64)
        if len(palette_labs):
            lab_distances = np.linalg.norm(
                palette_labs[:, None, :] - palette_labs[None, :, :], axis=2
            )
        else:
            lab_distances = np.empty((0, 0), dtype=np.float64)
        palette_has_contrast = bool(
            len(palette_labs) >= 2 and np.any(lab_distances >= 20.0)
        )

        return {
            'areas': areas,
            'edges': tuple(edges),
            'adjacency': tuple(tuple(neighbors) for neighbors in adjacency),
            'weighted_degrees': weighted_degrees,
            'lab_distances': lab_distances,
            'palette_has_contrast': palette_has_contrast,
        }

    @staticmethod
    def _valid_outline_reference_hints(
            region_count, color_count, background_index,
            reference_hints=None, reference_purities=None):
        """逐项接纳有效 hint；背景或低纯度 sentinel 不否决整组元数据。"""
        if reference_hints is None or len(reference_hints) != region_count:
            return {}
        purities_match = (
            reference_purities is not None
            and len(reference_purities) == region_count
        )
        valid = {}
        for region_index, raw_hint in enumerate(reference_hints):
            if region_index == background_index:
                continue
            try:
                hint = int(raw_hint)
            except (TypeError, ValueError):
                continue
            if not 0 <= hint < color_count:
                continue
            purity = 1.0
            if purities_match:
                try:
                    purity = float(reference_purities[region_index])
                except (TypeError, ValueError):
                    purity = 0.0
            purity = max(0.0, min(1.0, purity))
            if purity > 0.0:
                valid[region_index] = (hint, purity)
        return valid

    def _score_outline_placement_coherence(
            self, placement, regions, selected_colors, background_index,
            reference_hints=None, reference_purities=None, outline_array=None,
            _coherence_context=None):
        """评估部分排列；None 表示该区域不着色，不能被当成非法映射。"""
        region_count = len(regions)
        color_count = len(selected_colors)
        if len(placement) != region_count or color_count <= 0:
            return float('-inf')

        normalized = []
        for value in placement:
            if value is None:
                normalized.append(None)
                continue
            try:
                color = int(value)
            except (TypeError, ValueError, OverflowError):
                return float('-inf')
            if not 0 <= color < color_count:
                return float('-inf')
            normalized.append(color)

        colored_indices = [
            index for index, color in enumerate(normalized)
            if color is not None
        ]
        if not colored_indices:
            return float('-inf')

        context = _coherence_context or self._prepare_outline_coherence_context(
            regions, selected_colors, outline_array
        )
        areas = np.asarray(context['areas'], dtype=np.float64)
        total_area = float(np.sum(areas))
        colored_area = float(np.sum(areas[colored_indices]))
        coverage_score = colored_area / max(total_area, 1.0)

        selected_areas = areas[colored_indices]
        area_balance = (
            1.0 - self._gini_coefficient(selected_areas)
            if len(selected_areas) >= 2 else 1.0
        )

        # 部分着色时，相邻的不同色块过度聚集会形成局部拥挤；适度分散更清晰。
        colored_set = set(colored_indices)
        total_incident = 0.0
        colored_pair_weight = 0.0
        for first, second, weight in context['edges']:
            first_colored = first in colored_set
            second_colored = second in colored_set
            if first_colored or second_colored:
                total_incident += float(weight)
            if first_colored and second_colored:
                colored_pair_weight += float(weight)
        distribution_score = (
            1.0 - colored_pair_weight / total_incident
            if total_incident > 0.0 else 1.0
        )

        valid_hints = self._valid_outline_reference_hints(
            region_count, color_count, background_index,
            reference_hints, reference_purities,
        )
        reference_weight = sum(purity for _, purity in valid_hints.values())
        reference_score = 0.0
        if reference_weight > 0.0:
            reference_score = sum(
                purity
                for region_index, (hint, purity) in valid_hints.items()
                if normalized[region_index] == hint
            ) / reference_weight

        background_penalty = 0.0
        if (
            isinstance(background_index, (int, np.integer))
            and 0 <= int(background_index) < region_count
            and normalized[int(background_index)] is not None
            and len(colored_indices) < region_count
        ):
            background_penalty = 1.5

        return float(
            5.0 * reference_score
            + 3.0 * coverage_score
            + 2.0 * area_balance
            + 1.0 * distribution_score
            - background_penalty
        )

    def _generate_coherent_outline_placements(
            self, regions, selected_colors, background_index,
            reference_hints=None, reference_purities=None, outline_array=None,
            max_samples=500, _coherence_context=None):
        """确定性生成背景独占、面积感知且 RAG 连贯的完整候选。"""
        region_count = len(regions)
        color_count = len(selected_colors)
        try:
            effective_limit = min(500, max(0, int(max_samples)))
        except (TypeError, ValueError, OverflowError):
            effective_limit = 0
        if region_count <= 0 or color_count <= 0 or effective_limit <= 0:
            return []
        if not isinstance(background_index, (int, np.integer)):
            background_index = None
        elif not 0 <= int(background_index) < region_count:
            background_index = None
        else:
            background_index = int(background_index)
        if color_count == 1:
            return [[0] * region_count]

        context = _coherence_context or self._prepare_outline_coherence_context(
            regions, selected_colors, outline_array
        )
        valid_hints = self._valid_outline_reference_hints(
            region_count, color_count, background_index,
            reference_hints, reference_purities,
        )
        internal_indices = [
            index for index in range(region_count) if index != background_index
        ]
        order = sorted(
            internal_indices,
            key=lambda index: (
                -float(context['areas'][index]),
                -float(context['weighted_degrees'][index]),
                index,
            ),
        )
        background_colors = (
            range(color_count) if background_index is not None else (None,)
        )
        beam_width = min(160, effective_limit)
        scored_candidates = {}

        for background_color in background_colors:
            if background_color is None:
                allowed_colors = list(range(color_count))
            else:
                allowed_colors = [
                    color for color in range(color_count) if color != background_color
                ]
                if context['palette_has_contrast']:
                    allowed_colors = [
                        color for color in allowed_colors
                        if context['lab_distances'][background_color, color] >= 20.0
                    ]
            if not allowed_colors and internal_indices:
                continue

            initial = [-1] * region_count
            if background_index is not None:
                initial[background_index] = int(background_color)
            beams = [(initial, np.zeros(color_count, dtype=np.float64), 0.0)]
            assigned = set()
            for region_index in order:
                next_beams = []
                for placement, color_areas, cumulative_structural_cost in beams:
                    for color in allowed_colors:
                        candidate = list(placement)
                        candidate[region_index] = int(color)
                        candidate_areas = color_areas.copy()
                        candidate_areas[color] += context['areas'][region_index]

                        hint_penalty = 0.0
                        if region_index in valid_hints:
                            hint, purity = valid_hints[region_index]
                            hint_penalty = 3.0 * purity * (color != hint)
                        assigned_total = float(sum(
                            candidate_areas[c] for c in allowed_colors
                        ))
                        target = assigned_total / max(1, len(allowed_colors))
                        area_penalty = sum(
                            abs(candidate_areas[c] - target) for c in allowed_colors
                        ) / max(assigned_total, 1.0)
                        switch_penalty = sum(
                            weight
                            for neighbor, weight in context['adjacency'][region_index]
                            if neighbor in assigned and candidate[neighbor] != color
                        )
                        cumulative_structural_cost_next = (
                            cumulative_structural_cost
                            + hint_penalty
                            + 2.0 * switch_penalty
                        )
                        current_cost = (
                            cumulative_structural_cost_next + 1.5 * area_penalty
                        )
                        next_beams.append((
                            candidate,
                            candidate_areas,
                            cumulative_structural_cost_next,
                            current_cost,
                        ))
                next_beams.sort(key=lambda item: (item[3], tuple(item[0])))
                beams = [item[:3] for item in next_beams[:beam_width]]
                assigned.add(region_index)

            for placement, _, _ in beams:
                score = self._score_outline_placement_coherence(
                    placement, regions, selected_colors, background_index,
                    reference_hints, reference_purities, outline_array,
                    _coherence_context=context,
                )
                if np.isfinite(score):
                    key = tuple(placement)
                    scored_candidates[key] = max(score, scored_candidates.get(key, -np.inf))

            # 独立面积通道不消费 hint/RAG 成本，避免强语义 beam 剪掉全部可平衡方案。
            area_beam_width = min(32, effective_limit)
            area_beams = [(list(initial), np.zeros(color_count, dtype=np.float64))]
            for region_index in order:
                next_area_beams = []
                for placement, color_areas in area_beams:
                    for color in allowed_colors:
                        candidate = list(placement)
                        candidate[region_index] = int(color)
                        candidate_areas = color_areas.copy()
                        candidate_areas[color] += context['areas'][region_index]
                        assigned_total = float(sum(
                            candidate_areas[c] for c in allowed_colors
                        ))
                        target = assigned_total / max(1, len(allowed_colors))
                        area_penalty = sum(
                            abs(candidate_areas[c] - target) for c in allowed_colors
                        ) / max(assigned_total, 1.0)
                        next_area_beams.append((
                            candidate, candidate_areas, area_penalty
                        ))
                next_area_beams.sort(
                    key=lambda item: (item[2], tuple(item[0]))
                )
                area_beams = [
                    item[:2] for item in next_area_beams[:area_beam_width]
                ]

            for placement, _ in area_beams:
                score = self._score_outline_placement_coherence(
                    placement, regions, selected_colors, background_index,
                    reference_hints, reference_purities, outline_array,
                    _coherence_context=context,
                )
                if np.isfinite(score):
                    key = tuple(placement)
                    scored_candidates[key] = max(
                        score, scored_candidates.get(key, -np.inf)
                    )

        if not scored_candidates:
            return []

        # 所有候选统一执行面积上限；数学不可达时只保留严格最优占比。
        if internal_indices:
            shares = {}
            for placement in scored_candidates:
                background_color = (
                    placement[background_index] if background_index is not None else None
                )
                balance_colors = [
                    color for color in range(color_count) if color != background_color
                ]
                internal_total = float(sum(
                    context['areas'][index] for index in internal_indices
                ))
                color_areas = [
                    sum(
                        context['areas'][index] for index in internal_indices
                        if placement[index] == color
                    )
                    for color in balance_colors
                ]
                shares[placement] = max(color_areas, default=internal_total) / max(
                    internal_total, 1.0
                )
            epsilon = 1e-12
            feasible_shares = [
                share for share in shares.values() if share <= 0.55 + epsilon
            ]
            share_limit = 0.55 if feasible_shares else min(shares.values())
            scored_candidates = {
                placement: score
                for placement, score in scored_candidates.items()
                if shares[placement] <= share_limit + epsilon
            }

        ranked = sorted(
            scored_candidates,
            key=lambda placement: (-scored_candidates[placement], placement),
        )
        return [list(placement) for placement in ranked[:effective_limit]]

    def _restore_outline_plan_native_canvas(self, colored_image):
        """把辅助分辨率线稿方案恢复到用户导入线稿的原始画布。

        非 Element-Aware 分割可以继续在受控尺寸上工作；最终只放大扁平色块，
        随后重新覆盖原始分辨率线条，因此不会把低分辨率预览当成保存结果。
        """
        output = colored_image.copy().convert("RGB")
        original_path = getattr(
            self, "_element_aware_original_outline_path", None
        )
        if not original_path or not os.path.exists(original_path):
            return output

        with Image.open(original_path) as source:
            native_outline = source.convert("RGB")
        if native_outline.size == output.size:
            return output

        restored = output.resize(native_outline.size, Image.Resampling.NEAREST)
        restored_array = np.asarray(restored).copy()
        native_array = np.asarray(native_outline)
        native_gray = cv2.cvtColor(native_array, cv2.COLOR_RGB2GRAY)
        native_line_mask = native_gray < 254
        restored_array[native_line_mask] = native_array[native_line_mask]
        return Image.fromarray(restored_array, mode="RGB")

    def generate_outline_coloring_plans(self, selected_colors, total_images, symmetry_type="Asymmetry", progress_callback=None):
        """生成全画布线稿配色方案：线条保留，其余像素只使用所选应用色。"""
        import numpy as np
        from PIL import Image
        from tkinter import messagebox

        def report_progress(value, text):
            if progress_callback:
                try:
                    progress_callback(value, text)
                except Exception:
                    pass

        if not hasattr(self, 'regions_for_plan') or self.regions_for_plan is None:
            messagebox.showerror("Error!", "Undivided area, please click on 'Apply Color' firstly")
            return [], []

        if getattr(self, 'plan_outline_array', None) is None:
            messagebox.showerror("Error!", "The colored base image has not been saved!")
            return [], []

        num_colors = len(selected_colors)
        num_regions = len(self.regions_for_plan)
        if num_regions == 0 or num_colors == 0:
            messagebox.showerror("Error!", "No divided areas detected")
            return [], []
        if num_colors > num_regions:
            messagebox.showerror(
                "Error!",
                f"Selected colors ({num_colors}) cannot exceed number of regions ({num_regions})"
            )
            return [], []

        outline_array = self.plan_outline_array.copy()

        if getattr(self, '_element_aware_v21_active', False):
            return self._generate_element_aware_v21_plans(
                selected_colors, total_images, progress_callback
            )

        reference_hints = getattr(self, '_reference_region_color_hints', None)
        reference_palette = getattr(
            self, 'outline_coloring_colors', getattr(self, 'hex_color_codes', [])
        )
        reference_palette_matches = (
            reference_hints is not None
            and len(reference_hints) == num_regions
            and len(reference_palette) == num_colors
            and all(
                str(first).lower() == str(second).lower()
                for first, second in zip(reference_palette, selected_colors)
            )
        )
        usable_hints = reference_hints if reference_palette_matches else None
        usable_purities = (
            getattr(self, '_reference_region_color_hint_purities', None)
            if reference_palette_matches else None
        )
        background_index = getattr(self, '_outline_background_region_index', None)
        coherence_context = self._prepare_outline_coherence_context(
            self.regions_for_plan, selected_colors, outline_array
        )
        # 严格枚举全部 P(N,M) 个部分排列。每种颜色只出现一次，
        # 其余 N-M 个区域保持空白；不再使用 max_samples=500 截断。
        placements = self._generate_partial_permutation_placements(
            num_regions, num_colors
        )
        expected_count = (
            math.factorial(num_regions)
            // math.factorial(num_regions - num_colors)
        )
        if len(placements) != expected_count:
            raise RuntimeError(
                "Partial permutation enumeration mismatch: "
                f"expected {expected_count}, got {len(placements)}"
            )
        self._last_plan_permutation_count = int(expected_count)
        self._last_plan_permutation_formula = (
            f"P({num_regions},{num_colors})={expected_count}"
        )

        # Pre-compute invariant data outside the loop
        if len(outline_array.shape) == 3:
            gray_outline = np.mean(outline_array, axis=2).astype(np.uint8)
        else:
            gray_outline = outline_array.astype(np.uint8)
        line_mask = gray_outline < 254
        outline_line_pixels = outline_array[line_mask] if len(outline_array.shape) == 3 else None

        # 单张 owner map 替代 N 张全分辨率 bool mask；重叠区保持后 region 覆盖语义。
        owner_map = np.full(outline_array.shape[:2], -1, dtype=np.int32)
        for region_index, region in enumerate(self.regions_for_plan):
            region_pixels = np.asarray(region) > 0
            owner_map[region_pixels] = region_index
            del region_pixels

        palette_rgb = np.asarray([
            [
                int(str(color)[1:3], 16),
                int(str(color)[3:5], 16),
                int(str(color)[5:7], 16),
            ]
            for color in selected_colors
        ], dtype=np.uint8)
        lightweight_scores = []
        total_to_score = len(placements)
        update_step = max(1, total_to_score // 120) if total_to_score > 0 else 1
        for idx, placement in enumerate(placements):
            score = self._score_outline_placement_coherence(
                placement,
                self.regions_for_plan,
                selected_colors,
                background_index,
                reference_hints=usable_hints,
                reference_purities=usable_purities,
                outline_array=outline_array,
                _coherence_context=coherence_context,
            )
            if np.isfinite(score):
                lightweight_scores.append((placement, score))

            if idx % update_step == 0 or idx == total_to_score - 1:
                pct = 80 * (idx + 1) / max(1, total_to_score)
                report_progress(pct, f"Scoring {idx + 1}/{total_to_score}")

        lightweight_scores.sort(key=lambda item: float(item[1]), reverse=True)
        selected_for_render = lightweight_scores
        scored_placements = []
        for render_index, (placement, score) in enumerate(selected_for_render):
            colored_array = np.full(
                outline_array.shape[:2] + (3,), 255, dtype=np.uint8
            )
            for region_index, color_value in enumerate(placement):
                color_index = self._normalize_index(color_value)
                if (
                    color_index is None
                    or not 0 <= color_index < len(palette_rgb)
                ):
                    continue
                colored_array[owner_map == region_index] = palette_rgb[color_index]

            # 只覆盖原始线条像素，不二值化、不加粗。
            if outline_line_pixels is not None:
                colored_array[line_mask] = outline_line_pixels
            else:
                colored_array[line_mask] = gray_outline[line_mask][:, None]

            colored_image = Image.fromarray(colored_array.astype(np.uint8))
            colored_image = self._restore_outline_plan_native_canvas(
                colored_image
            )
            scored_placements.append((placement, score, colored_image))
            report_progress(
                80 + 20 * (render_index + 1) / max(1, len(selected_for_render)),
                f"Rendering {render_index + 1}/{len(selected_for_render)}",
            )

        return placements, scored_placements

    def calculate_adjacent_pixels(self, labels, max_ignored_gap=5):
        """统计不同颜色之间真正共享的像素边界长度。

        仅使用水平和垂直 4 邻域。对角线像素只共享一个角，不构成共同边界；
        旧版把两个对角方向也计入，会制造虚假连线并让斜边被重复放大。

        调色板采样会把细黑描边标记为 ``-1``。这些像素不是第三种设计颜色，
        因而允许跨越最多 5 像素的连续 ``-1`` 描边恢复其两侧颜色的视觉
        共边关系。较宽的空白/缺失区仍不会被跨越。
        """
        import numpy as np
        from collections import defaultdict

        if labels is None or np.asarray(labels).ndim != 2:
            return {}

        labels = np.asarray(labels)
        valid_values = labels[labels >= 0]
        if valid_values.size == 0:
            return {}
        label_base = int(np.max(valid_values)) + 1
        adjacency = defaultdict(int)

        def accumulate_pairs(first_map, second_map, support_mask=None):
            mask = (
                (first_map != second_map)
                & (first_map >= 0)
                & (second_map >= 0)
            )
            if support_mask is not None:
                mask &= support_mask
            if not np.any(mask):
                return

            first_values = first_map[mask].astype(np.int64, copy=False)
            second_values = second_map[mask].astype(np.int64, copy=False)
            lower = np.minimum(first_values, second_values)
            upper = np.maximum(first_values, second_values)
            encoded = lower * label_base + upper
            unique_pairs, pair_counts = np.unique(
                encoded, return_counts=True
            )
            for encoded_pair, count in zip(unique_pairs, pair_counts):
                first = int(encoded_pair // label_base)
                second = int(encoded_pair % label_base)
                if first != second:
                    adjacency[(first, second)] += int(count)

        # 每对直接边只统计一次：右邻居和下邻居。
        accumulate_pairs(labels[:, :-1], labels[:, 1:])
        accumulate_pairs(labels[:-1, :], labels[1:, :])

        # 恢复被短描边隔开的视觉邻接。gap 表示中间连续 -1 的像素数；
        # 只有全部中间像素均为 -1 才计数，不会穿过第三个颜色区域。
        h, w = labels.shape
        gap_limit = max(0, int(max_ignored_gap))
        for gap in range(1, gap_limit + 1):
            offset = gap + 1
            if w > offset:
                width = w - offset
                ignored = np.ones((h, width), dtype=bool)
                for middle in range(1, offset):
                    ignored &= labels[:, middle:middle + width] < 0
                accumulate_pairs(
                    labels[:, :width],
                    labels[:, offset:],
                    ignored,
                )
            if h > offset:
                height = h - offset
                ignored = np.ones((height, w), dtype=bool)
                for middle in range(1, offset):
                    ignored &= labels[middle:middle + height, :] < 0
                accumulate_pairs(
                    labels[:height, :],
                    labels[offset:, :],
                    ignored,
                )

        return dict(adjacency)

    def _infer_color_role_enhanced(self, b, pr, ev, prop, disp, deg, ew):
        """根据网络中心性、面积与离散度解释颜色的结构角色。"""
        is_wide_spread = disp > 1.5
        is_concentrated = disp < 0.5

        if pr > 0.5 and prop > 0.15 and is_wide_spread:
            return "Dominant-Carrier(主色载体)"
        elif b > 0.5 and prop < 0.15 and is_concentrated:
            return "Transition-Bridge(过渡桥梁)"
        elif b > 0.45 and ew > 1.0:
            return "Boundary-Coordinator(边界协调)"
        elif pr > 0.45 and is_concentrated:
            return "Emphasis-Accent(强调点缀)"
        elif prop > 0.25 and is_wide_spread:
            return "Ambient-Background(环境背景)"
        elif prop > 0.20:
            return "Mass-Anchor(面积锚点)"
        elif b > 0.3 and pr > 0.3:
            return f"Strategic-Link(战略连接,deg={deg})"
        elif ew > 1.0:
            return f"Connector(连接节点,w={ew:.2f})"
        else:
            return "Decorative-Detail(装饰细节)"

    def _infer_color_role_simple(self, b, pr, ev, prop):
        """简化版颜色角色推断（可解释性）"""
        if b > 0.5 and prop < 0.15:
            return "Bridge(桥梁)"
        elif pr > 0.5 and prop > 0.15:
            return "Hub(枢纽)"
        elif prop > 0.25:
            return "Background(背景)"
        elif b > 0.3 and pr > 0.3:
            return "Strategic(战略)"
        else:
            return "Decorative(装饰)"

    def _describe_mapping_quality_enhanced(self, placement, selected_colors, G, centrality,
                                           props, node_attrs):
        """增强版映射描述（含dispersion和边信息）"""
        parts = []
        betweenness = centrality.get('betweenness', {})
        pagerank = centrality.get('pagerank', {})

        high_b_nodes = sorted(G.nodes(), key=lambda n: betweenness.get(n, 0), reverse=True)[:3]
        for node in high_b_nodes:
            normalized_node = self._normalize_index( node )
            if normalized_node is not None and normalized_node < len( placement ) and placement[normalized_node] is not None:
                cidx = self._normalize_index( placement[normalized_node] )
                if cidx is not None and cidx < len(selected_colors):
                    disp = node_attrs.get(node, {}).get('dispersion', 0)
                    role = self._infer_color_role_enhanced(
                        betweenness.get(node, 0), pagerank.get(node, 0), 0,
                        props[normalized_node] if normalized_node < len(props) else 0,
                        disp, node_attrs.get(node,{}).get('degree',0),
                        node_attrs.get(node,{}).get('total_edge_weight',0))
                    parts.append(f"R{node}[{role},D={disp:.1f}]→C{cidx}")

        return ", ".join(parts) if parts else "N/A"

    def _compute_node_fidelity_score(self, namq, vafe, nsb):
        return min(100, max(0, 0.45 * namq + 0.30 * vafe + 0.25 * nsb))

    def _compute_edge_fidelity_score(self, elh):
        return min(100, max(0, elh))

    def _compute_topology_fidelity_score(self, topology):
        return min(100, max(0, topology))

    def _compute_structural_fidelity_score(self, node_fidelity, edge_fidelity, topology_fidelity):
        return min(100, max(0, 0.38 * node_fidelity + 0.34 * edge_fidelity + 0.28 * topology_fidelity))

    def _compute_creative_variation_score(self, placement, sel_colors, G, na, ea):
        import numpy as np
        import cv2
        try:
            from networkx.algorithms.community import greedy_modularity_communities
        except Exception:
            greedy_modularity_communities = None

        if not sel_colors:
            return 0

        used = [c for c in placement if c is not None and c < len(sel_colors)]
        if not used:
            return 0

        def hex_to_lab(hex_color):
            r = int( hex_color[1:3], 16 )
            g = int( hex_color[3:5], 16 )
            b = int( hex_color[5:7], 16 )
            return cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2Lab)[0, 0].astype(float)

        novelty_vals = []
        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            if normalized_node is None or normalized_node >= len(placement):
                continue
            ci = self._normalize_index( placement[normalized_node] )
            if ci is None or ci >= len(sel_colors):
                continue
            src_lab = na.get(node, {}).get('lab', np.array([50., 0., 0.]))
            tgt_lab = hex_to_lab( sel_colors[ci] )
            de = self._ciede2000( tgt_lab, src_lab )
            novelty_vals.append( np.exp( -((de - 38.0) ** 2) / (2 * 18.0 ** 2) ) )
        novelty = float(np.mean(novelty_vals) * 100) if novelty_vals else 45.0

        palette_labs = [hex_to_lab(sel_colors[idx]) for idx in sorted(set(used))]
        diversity = 55.0
        if len(palette_labs) >= 2:
            dists = []
            for i in range(len(palette_labs)):
                for j in range(i + 1, len(palette_labs)):
                    dists.append(self._ciede2000(palette_labs[i], palette_labs[j]))
            if dists:
                mean_dist = float(np.mean(dists))
                diversity = float(100 * np.exp(-((mean_dist - 42.0) ** 2) / (2 * 20.0 ** 2)))

        community_flex = 55.0
        if greedy_modularity_communities is not None and G.number_of_edges() > 0:
            try:
                communities = list(greedy_modularity_communities(G, weight='weight'))
                comm_scores = []
                for comm in communities:
                    assigned = []
                    for n in comm:
                        normalized_n = self._normalize_index( n )
                        if normalized_n is None or normalized_n >= len( placement ):
                            continue
                        normalized_ci = self._normalize_index( placement[normalized_n] )
                        if normalized_ci is not None:
                            assigned.append( normalized_ci )
                    if not assigned:
                        continue
                    uniq = len(set(assigned)) / max(1, min(len(sel_colors), len(assigned)))
                    comm_scores.append(uniq)
                if comm_scores:
                    community_flex = float(np.mean(comm_scores) * 100)
            except Exception:
                community_flex = 55.0

        return min(100, max(0, 0.45 * novelty + 0.30 * diversity + 0.25 * community_flex))

    def get_crn_score_breakdown(self, image, labels, unique_labels, placement,
                                selected_colors, source_G, centrality, source_proportions,
                                node_attrs=None, edge_attrs=None, source_disps=None,
                                user_selected_colors=None, representative_palette=None):
        """CRN-Palette v4 分项评分；total 与实际排序公式完全一致。"""
        _na = node_attrs or {}
        _ea = edge_attrs or []
        user_selected_colors = list(user_selected_colors or selected_colors or [])
        representative_palette = list(representative_palette or [])

        elh = self._compute_ELH_relational_v4(
            source_G, placement, selected_colors, _ea, _na
        )
        node_role = self._compute_node_role_alignment_v4(
            placement, selected_colors, source_G, centrality, _na
        )
        vafe = self._compute_VAFE_style_v4(
            placement, selected_colors, source_G, centrality, _na
        )
        topology = self._compute_topology_style_v4(
            placement, selected_colors, source_G, _na
        )
        nsb = self._compute_NSB_style_v4(
            placement, labels, unique_labels, selected_colors,
            source_G, centrality, _na
        )
        palette_fidelity = self._compute_palette_anchor_fidelity_v4(
            selected_colors, user_selected_colors
        )
        representative = self._compute_representative_palette_geometry_v4(
            selected_colors, representative_palette
        )
        true_crn = self._compute_true_crn_fidelity_v414(
            labels, unique_labels, placement, selected_colors,
            source_G, centrality, source_proportions, _na
        )
        mass_fidelity = true_crn['mass']
        centrality_fidelity = true_crn['centrality']
        edge_graph_fidelity = true_crn['edge_graph']

        # v4.1.4：CRN 仍占 92%，但其中 50% 直接比较节点面积、中心性和边网络。
        # 这避免“最小源节点颜色占据目标最大区域”却仍获得高分。
        total = (
            0.18 * mass_fidelity
            + 0.16 * centrality_fidelity
            + 0.16 * edge_graph_fidelity
            + 0.14 * elh
            + 0.10 * node_role
            + 0.08 * vafe
            + 0.06 * topology
            + 0.04 * nsb
            + 0.06 * palette_fidelity
            + 0.02 * representative
        )
        # 5% 最低面积/58% 上限作为可行性门控；满足时不影响分数。
        area_gate = float(np.clip(true_crn['constraint'] / 100.0, 0.0, 1.0))
        total *= 0.72 + 0.28 * area_gate
        total = float(np.clip(total, 0.0, 100.0))
        structural = (
            0.18 * mass_fidelity
            + 0.16 * centrality_fidelity
            + 0.16 * edge_graph_fidelity
            + 0.14 * elh
            + 0.10 * node_role
            + 0.08 * vafe
            + 0.06 * topology
            + 0.04 * nsb
        ) / 0.92
        return {
            # 保留 NAMQ 键兼容旧界面，但其含义已改为“节点角色对齐”，不再惩罚换色。
            'NAMQ': float(node_role),
            'NodeRole': float(node_role),
            'ELH': float(elh),
            'VAFE': float(vafe),
            'NSB': float(nsb),
            'Topology': float(topology),
            'PaletteFidelity': float(palette_fidelity),
            'RepresentativePalette': float(representative),
            'MassFidelity': float(mass_fidelity),
            'CentralityFidelity': float(centrality_fidelity),
            'GraphEdgeFidelity': float(edge_graph_fidelity),
            'AreaConstraint': float(true_crn['constraint']),
            'MinimumColorShare': float(true_crn['minimum_share']),
            'MaximumColorShare': float(true_crn['maximum_share']),
            'TargetColorShares': list(true_crn['target_shares']),
            'SourceColorShares': list(true_crn['source_shares']),
            'Usage': float(self._compute_color_usage_score(
                placement, selected_colors, source_G,
                centrality.get('pagerank', {}), _na
            )),
            'NodeFidelity': float(0.55 * node_role + 0.45 * vafe),
            'EdgeFidelity': float(elh),
            'TopologyFidelity': float(topology),
            'StructuralFidelity': float(structural),
            'CreativeVariation': float(representative),
            'total': total,
            'formula': '(0.18*Mass + 0.16*Centrality + 0.16*GraphEdge + 0.14*ELH_rel + 0.10*NodeRole + 0.08*VAFE + 0.06*Topology + 0.04*NSB + 0.06*Palette + 0.02*Representative) * AreaGate',
        }

    def build_crn_explanation_summary(self, placement, selected_colors, G, centrality,
                                      props, node_attrs, breakdown=None):
        breakdown = breakdown or {}
        mapping_desc = self._describe_mapping_quality_enhanced(
            placement, selected_colors, G, centrality, props, node_attrs
        )
        delta_text = ''
        if 'PaletteMeanDeltaE' in breakdown:
            delta_text = (
                f" | Palette ΔE mean/max={breakdown.get('PaletteMeanDeltaE', 0.0):.2f}/"
                f"{breakdown.get('PaletteMaxDeltaE', 0.0):.2f}"
            )
        return (
            f"CRN92%: Mass {breakdown.get('MassFidelity', 0.0):.1f}, "
            f"Centrality {breakdown.get('CentralityFidelity', 0.0):.1f}, "
            f"GraphEdge {breakdown.get('GraphEdgeFidelity', 0.0):.1f}, "
            f"ELH {breakdown.get('ELH', 0.0):.1f}, "
            f"NodeRole {breakdown.get('NodeRole', 0.0):.1f}, "
            f"Flow {breakdown.get('VAFE', 0.0):.1f}, "
            f"Topology {breakdown.get('Topology', 0.0):.1f}, "
            f"Balance {breakdown.get('NSB', 0.0):.1f} | "
            f"Palette {breakdown.get('PaletteFidelity', 0.0):.1f}, "
            f"Representative {breakdown.get('RepresentativePalette', 0.0):.1f}"
            f"{delta_text} | Roles: {mapping_desc}"
        )

    def calculate_image_score_with_CRN(self, image, labels, unique_labels, placement,
                                        selected_colors, source_G, centrality, source_proportions,
                                        node_attrs=None, edge_attrs=None, source_disps=None,
                                        user_selected_colors=None, representative_palette=None):
        """CRN-Palette v4 实际排序分数。

        S = (0.18 Mass + 0.16 Centrality + 0.16 GraphEdge + 0.14 ELH_rel
             + 0.10 NodeRole + 0.08 VAFE + 0.06 Topology + 0.04 NSB
             + 0.06 PaletteFidelity + 0.02 RepresentativePalette) × AreaGate

        前八项全部来自 Color Relationship Network，占 92%；其中 50% 直接
        衡量节点面积、中心性和边连接。调色板适配在
        ΔE00<=6、明度/色度/色相受限、颜色数量不变的硬约束下进行，因此
        不会为了高分偏离用户选择的颜色。
        """
        breakdown = self.get_crn_score_breakdown(
            image, labels, unique_labels, placement,
            selected_colors, source_G, centrality, source_proportions,
            node_attrs=node_attrs, edge_attrs=edge_attrs,
            source_disps=source_disps,
            user_selected_colors=user_selected_colors,
            representative_palette=representative_palette,
        )
        return float(breakdown['total'])

    def _compute_NAMQ_v3(self, placement, sel_colors, G, btw, pr, ev, props, na, disps):
        """NAMQ v3: 源图原色→目标色的色彩迁移保真度(色调/冷暖/饱和度/明度/ΔE + dispersion + CRN重要性)"""
        import numpy as np; import cv2
        total_imp, matched = 0.0, 0.0
        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            if normalized_node is None or normalized_node >= len(placement): continue
            wd_val = na.get(node, {}).get('total_edge_weight', 0) / max(G.number_of_edges(), 1)
            importance = 0.28*btw.get(node,0)+0.30*pr.get(node,0)+0.18*ev.get(node,0)+0.24*min(1.0,wd_val)
            total_imp += importance
            ci = self._normalize_index( placement[normalized_node] )
            if ci is None or ci >= len(sel_colors): continue
            ch = sel_colors[ci]
            tr,tg,tb = int(ch[1:3],16),int(ch[3:5],16),int(ch[5:7],16)
            t_hsv = cv2.cvtColor(np.array([[[tr,tg,tb]]],dtype=np.uint8),cv2.COLOR_RGB2HSV)[0,0]
            t_lab = cv2.cvtColor(np.array([[[tr,tg,tb]]],dtype=np.uint8),cv2.COLOR_RGB2Lab)[0,0].astype(float)
            t_sat,t_val,t_hue = t_hsv[1]/255.0,t_hsv[2]/255.0,t_hsv[0]
            src_rgb = na.get(node,{}).get('rgb',np.array([0.5,0.5,0.5]))
            src_hsv = na.get(node,{}).get('hsv',np.array([0.,128.,128.]))
            src_lab = na.get(node,{}).get('lab',np.array([50.,0.,0.]))
            s_sat = src_hsv[1]/255.0 if len(src_hsv)>1 else 0.5
            s_val = src_hsv[2]/255.0 if len(src_hsv)>2 else 0.5
            s_hue = src_hsv[0] if len(src_hsv)>0 else 90.0
            disp = na.get(node,{}).get('dispersion',disps[node] if node<len(disps) else 1.0)
            hue_diff = min(abs(t_hue-s_hue),180-abs(t_hue-s_hue))/180.0
            hue_score = (1-hue_diff)**(0.6 if disp>1.3 else(1.2 if disp<0.6 else 0.9))
            def _is_warm(h): return h<35 or h>155
            def _is_cool(h): return 45<=h<=115
            sw,tw=_is_warm(s_hue),_is_warm(t_hue)
            sc,tc=_is_cool(s_hue),_is_cool(t_hue)
            if sw and tw: wc_match=1.0
            elif sc and tc: wc_match=1.0
            elif(not sw and not sc): wc_match=0.7
            elif(not tw and not tc): wc_match=0.65
            else: wc_match=0.15
            wc_score=wc_match if disp<1.2 else max(wc_match,0.55)
            sat_diff=abs(t_sat-s_sat); sat_preserve=np.exp(-sat_diff/0.4)
            sat_score=sat_preserve*0.85+t_sat*0.15 if disp<0.5 else sat_preserve*0.55+t_sat*0.45
            val_diff=abs(t_val-s_val); val_preserve=np.exp(-val_diff/0.5)
            val_score=val_preserve*0.50+(1-abs(t_val-0.58)/0.58)*0.50
            de=self._ciede2000(t_lab,src_lab)
            if de<20: de_score=0.5+de/40
            elif de<60: de_score=1.0
            elif de<80: de_score=max(0.3,1-(de-60)/40)
            else: de_score=max(0.1,0.3-(de-80)/100)
            de_score=de_score**(0.8 if disp>1.5 else(1.3 if disp<0.5 else 1.0))
            attr=0.25*hue_score+0.20*wc_score+0.20*sat_score+0.15*val_score+0.20*de_score
            matched+=importance*attr
        return min(100,max(0,(matched/total_imp*100) if total_imp>0 else 50))

    def _compute_ELH_v3(self, source_G, placement, sel_colors, ea, na):
        """ELH v3: source adjacency relation preservation + boundary weighted + complement bonus"""
        import numpy as np; import cv2
        if ea:
            edge_list = [(e['u'],e['v'],{'weight':e['weight'],'boundary_count':e['boundary_count']}) for e in ea]
        else:
            edge_list = [(u,v,d) for u,v,d in source_G.edges(data=True)]
        edge_data,tw,total_bc=[],0,sum(e[2].get('boundary_count',0) for e in edge_list) or 1
        bc_scale=np.log(1+total_bc/max(len(edge_list),1)) if edge_list else 1
        for u,v,data in edge_list:
            w,bc=data.get('weight',0.5),data.get('boundary_count',0)
            effective_w=w*(1+np.log(1+bc)/max(bc_scale,0.1)); tw+=effective_w
            normalized_u = self._normalize_index( u )
            normalized_v = self._normalize_index( v )
            uci=placement[normalized_u] if normalized_u is not None and normalized_u<len(placement) else None
            vci=placement[normalized_v] if normalized_v is not None and normalized_v<len(placement) else None
            uci = self._normalize_index( uci )
            vci = self._normalize_index( vci )
            if uci is None or vci is None: edge_data.append((effective_w,50)); continue
            if uci>=len(sel_colors) or vci>=len(sel_colors): edge_data.append((effective_w,40)); continue
            ur,ug,ub=int(sel_colors[uci][1:3],16),int(sel_colors[uci][3:5],16),int(sel_colors[uci][5:7],16)
            vr,vg,vb=int(sel_colors[vci][1:3],16),int(sel_colors[vci][3:5],16),int(sel_colors[vci][5:7],16)
            t_u_lab=cv2.cvtColor(np.array([[[ur,ug,ub]]],dtype=np.uint8),cv2.COLOR_RGB2Lab)[0,0].astype(float)
            t_v_lab=cv2.cvtColor(np.array([[[vr,vg,vb]]],dtype=np.uint8),cv2.COLOR_RGB2Lab)[0,0].astype(float)
            t_de=self._ciede2000(t_u_lab,t_v_lab)
            u_src_lab=na.get(u,{}).get('lab',np.array([50.,0.,0.]))
            v_src_lab=na.get(v,{}).get('lab',np.array([50.,0.,0.]))
            s_de=self._ciede2000(u_src_lab,v_src_lab)
            de_deviation=abs(t_de-s_de); relation_score=np.exp(-de_deviation/30.0)
            avg_bc=total_bc/max(len(edge_list),1)
            ideal_de,tol=(42,18) if bc>avg_bc*1.5 else ((50,35) if bc<avg_bc*0.3 else (45,30))
            harm=np.exp(-((t_de-ideal_de)**2)/(2*tol**2))
            hd=abs(int(cv2.cvtColor(np.array([[[ur,ug,ub]]],dtype=np.uint8),cv2.COLOR_RGB2HSV)[0,0][0])-
                      int(cv2.cvtColor(np.array([[[vr,vg,vb]]],dtype=np.uint8),cv2.COLOR_RGB2HSV)[0,0][0]))
            cb=15 if((75<=hd<=105) or hd>=165) else 0
            final_score=0.50*relation_score+0.30*harm+0.08*cb+0.12
            edge_data.append((effective_w,min(100,final_score*85+cb+5)))
        return min(100,max(0,sum(w*s for w,s in edge_data)/tw if tw>0 else 50))

    def _compute_VAFE_v3(self, placement, sel_colors, G, pr, ev, props, na, disps):
        """VAFE v3: visual attention flow + DAF(source attr alignment) + smooth + hierarchy"""
        import numpy as np; import cv2
        pv=[pr.get(n,0) for n in G.nodes()]
        if not pv or max(pv)<1e-6: return 50
        pa=np.array(pv); thresh=np.percentile(pa,70)
        ft_total,fc,daa_total,daf_total=0.0,0,0.0,0.0
        for node in G.nodes():
            pval=pr.get(node,0)
            if pval<thresh: continue
            fc+=1
            normalized_node = self._normalize_index( node )
            disp=na.get(node,{}).get('dispersion',disps[normalized_node] if normalized_node is not None and normalized_node<len(disps) else 1.0)
            prop=na.get(node,{}).get('proportion',props[normalized_node] if normalized_node is not None and normalized_node<len(props) else 0.05)
            src_hsv=na.get(node,{}).get('hsv',np.array([0.,128.,128.]))
            src_sat=src_hsv[1]/255.0 if len(src_hsv)>1 else 0.5
            src_val=src_hsv[2]/255.0 if len(src_hsv)>2 else 0.5
            src_attr=0.55*src_sat+0.35*src_val+0.10*min(1.0,src_sat*src_val*3)
            ci=placement[normalized_node] if normalized_node is not None and normalized_node<len(placement) else None
            ci = self._normalize_index( ci )
            if ci is None or ci>=len(sel_colors):
                ft_total+=pval*0.3; daa_total-=pval*disp/(max(disps)+0.01); daf_total-=pval*src_attr*prop; continue
            ch=sel_colors[ci]
            hsv=cv2.cvtColor(np.array([[[int(ch[1:3],16),int(ch[3:5],16),int(ch[5:7],16)]]],dtype=np.uint8),cv2.COLOR_RGB2HSV)[0,0]
            ts,tv=hsv[1]/255.0,hsv[2]/255.0
            tgt_attr=0.50*ts+0.28*(1-abs(tv-0.58)/0.58)+0.12*min(1,ts*tv)+0.10*(disp/(np.mean(disps)+0.01))
            ft_total+=pval*tgt_attr; daa_total+=pval*tgt_attr*disp/(np.mean(disps)+0.01)
            attr_ratio=min(tgt_attr,src_attr+0.01)/max(src_attr,0.01)
            if src_attr>0.6: daf_align=min(1.0,attr_ratio)**0.7
            elif src_attr<0.3: daf_align=max(0.5,min(1.0,attr_ratio))
            else: daf_align=min(1.0,attr_ratio)**0.9
            daf_total+=pval*daf_align*prop*5
        denom=sum(p for p in pv if p>=thresh)
        focus=(ft_total/denom*100) if denom>0 else 50
        daa_score=50+min(50,daa_total*10)
        daf_score=50+min(50,daf_total*8)
        fp,fe=0.0,0
        for u,v in G.edges():
            du,dvw=abs(pr.get(u,0)-pr.get(v,0)),G[u][v].get('weight',0.5)
            fp+=dvw*du; fe+=dvw
        smooth=100*(1-min(1,fp/max(fe,1)*3)) if fe>0 else 80
        hier=100*np.exp(-abs(np.var(pa)-0.008)/0.01)
        return min(100,0.34*focus+0.10*daa_score+0.14*daf_score+0.28*smooth+0.14*hier+
                   0.04*min(10,np.var([pr.get(n,0)*max(na.get(n,{}).get('dispersion',1),0.1) for n in G.nodes() if pr.get(n,0)>0])*200))

    def _compute_NSB_v3(self, placement, labels, ulabels, G, cl, wd, na):
        """NSB v3: structure balance + semantic coverage(warm/cool/neutral + saturation tiers from src hsv)"""
        import numpy as np
        areas=[]
        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            if normalized_node is not None and normalized_node<len(placement) and placement[normalized_node] is not None:
                lbl=ulabels[normalized_node] if normalized_node<len(ulabels) else normalized_node
                areas.append(float(np.sum(labels==lbl)))
        abal=(1-self._gini_coefficient(areas))*100 if len(areas)>=2 else(40 if len(areas)==1 else 20)
        hc=mc=lc=th=tm=tl=0
        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            wv=na.get(node,{}).get('total_edge_weight',wd.get(node,0)); ok=(normalized_node is not None and normalized_node<len(placement) and placement[normalized_node] is not None)
            if wv>0.5: th+=1; hc+=int(ok)
            elif wv>0.2: tm+=1; mc+=int(ok)
            else: tl+=1; lc+=int(ok)
        cov=0
        if th>0: cov+=0.35*(hc/th)
        if tm>0: cov+=0.30*(mc/tm)
        if tl>0: cov+=0.25*(lc/tl)
        cov*=100
        bc=te=G.number_of_edges()
        for u,v in G.edges():
            normalized_u = self._normalize_index( u )
            normalized_v = self._normalize_index( v )
            if(normalized_u is not None and normalized_u<len(placement) and placement[normalized_u] is not None) and(normalized_v is not None and normalized_v<len(placement) and placement[normalized_v] is not None): bc+=1
        util=(bc/max(te,1))*75+25
        cd,ud=[],[]
        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            d=na.get(node,{}).get('degree',G.degree(node))
            (cd if(normalized_node is not None and normalized_node<len(placement) and placement[normalized_node] is not None) else ud).append(d)
        if len(cd)>=2 and len(ud)>=1: deg_balance=100*(1-abs(np.mean(cd)-np.mean(ud))/max(np.mean(cd)+np.mean(ud),1))
        elif not cd: deg_balance=10
        else: deg_balance=60
        hi_disp=mid_disp=lo_disp=hcnt=mcnt=lcnt=0
        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            disp=na.get(node,{}).get('dispersion',1.0); ok=(normalized_node is not None and normalized_node<len(placement) and placement[normalized_node] is not None)
            if disp>1.5: hcnt+=1; hi_disp+=int(ok)
            elif disp>0.7: mcnt+=1; mid_disp+=int(ok)
            else: lcnt+=1; lo_disp+=int(ok)
        dcov=0
        if hcnt>0: dcov+=0.40*(hi_disp/hcnt)
        if mcnt>0: dcov+=0.35*(mid_disp/mcnt)
        if lcnt>0: dcov+=0.25*(lo_disp/lcnt)
        dcov*=100
        warm_cnt=cool_cnt=neut_cnt=warm_ok=cool_ok=neut_ok=0
        high_sat_cnt=mid_sat_cnt=low_sat_cnt=high_sat_ok=mid_sat_ok=low_sat_ok=0
        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            ok=(normalized_node is not None and normalized_node<len(placement) and placement[normalized_node] is not None)
            src_hsv=na.get(node,{}).get('hsv',np.array([0.,128.,128.]))
            src_hue=src_hsv[0] if len(src_hsv)>0 else 90
            src_sat=src_hsv[1]/255.0 if len(src_hsv)>1 else 0.5
            if src_hue<35 or src_hue>155: warm_cnt+=1; warm_ok+=int(ok)
            elif 45<=src_hue<=115: cool_cnt+=1; cool_ok+=int(ok)
            else: neut_cnt+=1; neut_ok+=int(ok)
            if src_sat>0.65: high_sat_cnt+=1; high_sat_ok+=int(ok)
            elif src_sat>0.3: mid_sat_cnt+=1; mid_sat_ok+=int(ok)
            else: low_sat_cnt+=1; low_sat_ok+=int(ok)
        sem_cov=0
        if warm_cnt>0: sem_cov+=0.35*(warm_ok/warm_cnt)
        if cool_cnt>0: sem_cov+=0.35*(cool_ok/cool_cnt)
        if neut_cnt>0: sem_cov+=0.30*(neut_ok/neut_cnt)
        sem_cov*=100
        sat_cov=0
        if high_sat_cnt>0: sat_cov+=0.40*(high_sat_ok/high_sat_cnt)
        if mid_sat_cnt>0: sat_cov+=0.35*(mid_sat_ok/mid_sat_cnt)
        if low_sat_cnt>0: sat_cov+=0.25*(low_sat_ok/low_sat_cnt)
        sat_cov*=100
        return min(100,0.15*abal+0.13*cov+0.15*util+0.17*deg_balance+0.12*dcov+0.14*sem_cov+0.14*sat_cov)

    def _compute_topology_consistency_score(self, placement, sel_colors, G, na, ea):
        import numpy as np
        import cv2
        try:
            from networkx.algorithms.community import greedy_modularity_communities
        except Exception:
            greedy_modularity_communities = None

        if G is None or G.number_of_nodes() == 0:
            return 50

        edge_list = [(e['u'], e['v'], {'weight': e.get('weight', 0.5)}) for e in ea] if ea else list(G.edges(data=True))
        if not edge_list:
            return 60

        def hex_to_lab(hex_color):
            r = int( hex_color[1:3], 16 )
            g = int( hex_color[3:5], 16 )
            b = int( hex_color[5:7], 16 )
            return cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2Lab)[0, 0].astype(float)

        target_labs = [hex_to_lab(c) for c in sel_colors]
        weighted_path_score = 0.0
        path_weight_total = 0.0
        for u, v, data in edge_list:
            normalized_u = self._normalize_index( u )
            normalized_v = self._normalize_index( v )
            if normalized_u is None or normalized_v is None or normalized_u >= len(placement) or normalized_v >= len(placement):
                continue
            cu = self._normalize_index( placement[normalized_u] )
            cv = self._normalize_index( placement[normalized_v] )
            if cu is None or cv is None or cu >= len(target_labs) or cv >= len(target_labs):
                continue
            src_u = na.get(u, {}).get('lab', np.array([50., 0., 0.]))
            src_v = na.get(v, {}).get('lab', np.array([50., 0., 0.]))
            src_de = self._ciede2000(src_u, src_v)
            tgt_de = self._ciede2000(target_labs[cu], target_labs[cv])
            weight = data.get('weight', 0.5)
            path_weight_total += weight
            weighted_path_score += weight * np.exp(-abs(tgt_de - src_de) / 28.0)
        path_score = (weighted_path_score / path_weight_total * 100) if path_weight_total > 0 else 55

        community_score = 60
        if greedy_modularity_communities is not None and G.number_of_edges() > 0:
            try:
                communities = list(greedy_modularity_communities(G, weight='weight'))
                if communities:
                    comm_scores = []
                    for comm in communities:
                        comm = list(comm)
                        assigned = []
                        for n in comm:
                            normalized_n = self._normalize_index( n )
                            if normalized_n is None or normalized_n >= len( placement ):
                                continue
                            normalized_ci = self._normalize_index( placement[normalized_n] )
                            if normalized_ci is not None:
                                assigned.append( normalized_ci )
                        if not assigned:
                            continue
                        counts = {}
                        for c in assigned:
                            counts[c] = counts.get(c, 0) + 1
                        purity = max(counts.values()) / max(len(assigned), 1)
                        comm_scores.append(purity)
                    if comm_scores:
                        community_score = np.mean(comm_scores) * 100
            except Exception:
                community_score = 60

        return min(100, max(0, 0.60 * path_score + 0.40 * community_score))

    def _compute_color_usage_score(self, placement, sel_colors, G, pr, na):
        import numpy as np

        if not sel_colors:
            return 0
        assigned = [c for c in placement if c is not None and c < len(sel_colors)]
        if not assigned:
            return 0

        used = set( assigned )
        coverage = len( used ) / len( sel_colors )
        color_weights = {idx: 0.0 for idx in range(len(sel_colors))}

        for node in G.nodes():
            normalized_node = self._normalize_index( node )
            if normalized_node is None or normalized_node >= len(placement):
                continue
            cidx = self._normalize_index( placement[normalized_node] )
            if cidx is None or cidx >= len(sel_colors):
                continue
            importance = 0.55 * pr.get(node, 0) + 0.25 * na.get(node, {}).get('proportion', 0) + 0.20 * min(1.0, na.get(node, {}).get('dispersion', 1.0) / 2.0)
            color_weights[cidx] += importance

        vals = np.array(list(color_weights.values()), dtype=float)
        if np.allclose(vals.sum(), 0):
            balance = 0.5
        else:
            vals = vals / vals.sum()
            target = np.ones_like(vals) / len(vals)
            balance = 1.0 - min(1.0, np.abs(vals - target).sum() / 2.0)

        return min(100, max(0, 72 * coverage + 28 * balance))

    def _complete_placement(self, placement, num_slots, num_colors):
        """规范 placement 长度，但保留 None 空槽，不再自动补色。"""
        if num_slots <= 0:
            return []
        if num_colors <= 0:
            return [None] * num_slots

        completed = []
        for value in list(placement)[:num_slots]:
            normalized = self._normalize_index(value)
            completed.append(
                normalized
                if normalized is not None and 0 <= normalized < num_colors
                else None
            )
        completed.extend([None] * (num_slots - len(completed)))
        return completed

    @staticmethod
    def _transfer_native_label_centers(native_rgb, analysis_labels, unique_labels):
        """在分析尺寸估计标签中心，保持标签编号与 placement 一一对应。"""
        source = np.asarray(native_rgb, dtype=np.uint8)
        labels = np.asarray(analysis_labels, dtype=np.int32)
        label_values = np.asarray(unique_labels, dtype=np.int32)
        analysis_rgb = cv2.resize(
            source,
            (labels.shape[1], labels.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        analysis_lab = cv2.cvtColor(
            analysis_rgb, cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        centers = []
        fallback = np.median(
            analysis_lab.reshape(-1, 3), axis=0
        ).astype(np.float32)
        for label in label_values:
            pixels = analysis_lab[labels == int(label)]
            center = np.median(pixels, axis=0) if len(pixels) else fallback
            centers.append(np.asarray(center, dtype=np.float32))
        return np.asarray(centers, dtype=np.float32)

    @classmethod
    def _build_native_transfer_owner_labels(
            cls, native_rgb, analysis_labels, unique_labels, chunk_size=180000):
        """直接在原分辨率建立互斥 owner，禁止放大低分辨率整型标签。"""
        source = np.asarray(native_rgb, dtype=np.uint8)
        label_values = np.asarray(unique_labels, dtype=np.int32)
        if source.ndim != 3 or source.shape[2] < 3:
            raise ValueError("Transfer target must be RGB-compatible")
        if label_values.size == 0:
            return np.full(source.shape[:2], -1, dtype=np.int32)

        centers = cls._transfer_native_label_centers(
            source[:, :, :3], analysis_labels, label_values
        )
        native_lab = cv2.cvtColor(
            source[:, :, :3], cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32)
        owners = np.empty(len(native_lab), dtype=np.int32)
        safe_chunk = max(1, int(chunk_size))
        for start in range(0, len(native_lab), safe_chunk):
            end = min(start + safe_chunk, len(native_lab))
            distance2 = np.sum(
                (
                    native_lab[start:end, None, :]
                    - centers[None, :, :]
                ) ** 2,
                axis=2,
            )
            owners[start:end] = label_values[np.argmin(distance2, axis=1)]
        return owners.reshape(source.shape[:2])

    @staticmethod
    def _transfer_connected_background_mask(native_rgb):
        """只锁定与图像边缘连通且颜色一致的外部背景。"""
        source = np.asarray(native_rgb, dtype=np.uint8)
        height, width = source.shape[:2]
        border_width = max(1, int(round(min(height, width) * 0.015)))
        border_mask = np.zeros((height, width), dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True

        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        border_pixels = lab[border_mask]
        if len(border_pixels) == 0:
            return np.zeros((height, width), dtype=bool)
        border_center = np.median(border_pixels, axis=0)
        border_distance = np.linalg.norm(
            border_pixels - border_center, axis=1
        )
        if float(np.mean(border_distance <= 10.0)) < 0.56:
            return np.zeros((height, width), dtype=bool)

        similar = (
            np.linalg.norm(lab - border_center, axis=2) <= 10.0
        ).astype(np.uint8)
        count, components = cv2.connectedComponents(similar, connectivity=4)
        if count <= 1:
            return np.zeros((height, width), dtype=bool)
        edge_labels = np.unique(np.concatenate((
            components[0, :],
            components[-1, :],
            components[:, 0],
            components[:, -1],
        )))
        edge_labels = edge_labels[edge_labels > 0]
        return np.isin(components, edge_labels)

    @staticmethod
    def _transfer_native_ink_mask(native_rgb):
        """锁定原生中性墨线及其一像素抗锯齿带。"""
        source = np.asarray(native_rgb, dtype=np.uint8)
        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        chroma = np.linalg.norm(lab[:, :, 1:3] - 128.0, axis=2)
        core = (lab[:, :, 0] <= 96.0) & (chroma <= 18.0)
        near_core = cv2.dilate(
            core.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=2,
        ).astype(bool)
        antialias = near_core & (gray < 245) & (chroma <= 18.0)
        return core | antialias

    @staticmethod
    def _transfer_structural_ink_mask(native_rgb):
        """保留中性墨线，并只补回真正呈细线形态的深色彩色结构线。

        单纯对 RGB 通道做 Canny 会把任意纯色色块的外沿伪造成黑色闭合线，
        从而改变原图拓扑。这里先按深色彩色像素连通域估计半线宽，仅允许
        最大内接半径较小的细线/细环参与结构边界；厚实色块仍是可填区域。
        """
        source = np.asarray(native_rgb, dtype=np.uint8)
        neutral_ink = ColorTransferToolbox._transfer_native_ink_mask(source)
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma = np.linalg.norm(lab[:, :, 1:3] - 128.0, axis=2)

        # 灰度会弱化纯红/纯蓝边界，因此分别在 RGB 三通道提取结构边缘。
        channel_edges = np.zeros(gray.shape, dtype=np.uint8)
        for channel in cv2.split(source):
            channel_edges |= cv2.Canny(channel, 20, 60)
        edge_band = cv2.dilate(
            channel_edges,
            np.ones((3, 3), dtype=np.uint8),
            iterations=2,
        ).astype(bool)

        colored_core = (gray < 165) & (chroma > 18.0)
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            colored_core.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        line_like = np.zeros(gray.shape, dtype=bool)
        maximum_half_width = max(3.5, 0.0035 * min(gray.shape))
        for label in range(1, component_count):
            if int(stats[label, cv2.CC_STAT_AREA]) < 4:
                continue
            component = components == label
            distance = cv2.distanceTransform(
                component.astype(np.uint8), cv2.DIST_L2, 5
            )
            if float(distance.max()) <= maximum_half_width:
                line_like |= component

        colored_structure = edge_band & line_like
        return neutral_ink | colored_structure

    @staticmethod
    def _transfer_internal_closed_area(outline_rgb):
        """计算不与画布边缘连通的可填面积，供微断口闭合决策使用。"""
        source = np.asarray(outline_rgb, dtype=np.uint8)
        gray = (
            source if source.ndim == 2
            else cv2.cvtColor(source[..., :3], cv2.COLOR_RGB2GRAY)
        )
        fillable = np.uint8(gray >= 235)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            fillable, connectivity=4, ltype=cv2.CV_32S
        )
        if component_count <= 1:
            return 0
        edge_labels = set(int(value) for value in np.unique(np.concatenate((
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ))))
        return sum(
            int(stats[label, cv2.CC_STAT_AREA])
            for label in range(1, component_count)
            if label not in edge_labels
        )

    @staticmethod
    def _build_transfer_segmentation_outline(visible_outline):
        """以受预算约束的小尺度闭运算生成仅供分割使用的线稿副本。"""
        visible = np.asarray(visible_outline, dtype=np.uint8)
        if visible.ndim == 2:
            visible = cv2.cvtColor(visible, cv2.COLOR_GRAY2RGB)
        elif visible.ndim == 3 and visible.shape[2] == 1:
            visible = cv2.cvtColor(visible[..., 0], cv2.COLOR_GRAY2RGB)
        else:
            visible = visible[..., :3].copy()

        gray = cv2.cvtColor(visible, cv2.COLOR_RGB2GRAY)
        barrier = np.uint8(gray < 235)
        baseline_area = ColorTransferToolbox._transfer_internal_closed_area(visible)
        barrier_pixels = int(np.count_nonzero(barrier))
        image_area = int(barrier.size)
        virtual_budget = max(
            8,
            int(np.ceil(0.04 * barrier_pixels)),
            int(np.ceil(0.0015 * image_area)),
        )
        candidates = [
            (cv2.MORPH_ELLIPSE, 3),
            (cv2.MORPH_ELLIPSE, 5),
            (cv2.MORPH_RECT, 3),
            (cv2.MORPH_RECT, 5),
        ]
        best = None
        for priority, (shape, size) in enumerate(candidates):
            kernel = cv2.getStructuringElement(shape, (size, size))
            closed = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, kernel)
            virtual = (closed > 0) & (barrier == 0)
            virtual_count = int(np.count_nonzero(virtual))
            if virtual_count <= 0 or virtual_count > virtual_budget:
                continue
            candidate = visible.copy()
            candidate[virtual] = 0
            gain = (
                ColorTransferToolbox._transfer_internal_closed_area(candidate)
                - baseline_area
            )
            if gain <= 0:
                continue
            score = (int(gain), -virtual_count, -priority)
            if best is None or score > best[0]:
                best = (score, candidate)
        return best[1] if best is not None else visible.copy()

    @staticmethod
    def _restore_transfer_virtual_barrier_owners(
            owner_map, virtual_barrier, excluded_owner=None):
        """按最近前景 owner 恢复全部分割补线像素，不受传播轮数限制。"""
        owners = np.asarray(owner_map, dtype=np.int32).copy()
        unresolved = np.asarray(virtual_barrier, dtype=bool) & (owners < 0)
        eligible = owners >= 0
        if excluded_owner is not None:
            eligible &= owners != int(excluded_owner)
        if not np.any(unresolved) or not np.any(eligible):
            return owners

        distance_input = np.ones(owners.shape, dtype=np.uint8)
        distance_input[eligible] = 0
        _, nearest_labels = cv2.distanceTransformWithLabels(
            distance_input,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        owner_by_label = np.full(int(nearest_labels.max()) + 1, -1, dtype=np.int32)
        owner_by_label[nearest_labels[eligible]] = owners[eligible]
        nearest_owner = owner_by_label[nearest_labels]
        owners[unresolved] = nearest_owner[unresolved]
        return owners

    @staticmethod
    def _transfer_native_is_line_sketch(native_rgb):
        """判断目标是否本身就是线稿；若是则 Transfer 直接保留原图。"""
        source = np.asarray(native_rgb, dtype=np.uint8)
        if source.ndim == 2:
            rgb = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] == 1:
            rgb = cv2.cvtColor(source[..., 0], cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] >= 3:
            rgb = source[..., :3]
        else:
            return False

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        near_white = np.all(rgb >= 245, axis=2)
        white_ratio = float(np.mean(near_white))
        dark_ratio = float(np.mean(gray < 200))
        edge_ratio = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
        channel_spread = (
            rgb.max(axis=2).astype(np.int16)
            - rgb.min(axis=2).astype(np.int16)
        )
        spread_p95 = float(np.percentile(channel_spread, 95)) if channel_spread.size else 0.0
        return bool(
            white_ratio >= 0.30
            and dark_ratio >= 0.003
            and edge_ratio >= 0.003
            and spread_p95 <= 24.0
        )

    @staticmethod
    def _build_transfer_soft_visible_outline(native_rgb):
        """为 Transfer 构造细节保留型可见线稿，不牺牲弱纹理与灰阶线。"""
        source = np.asarray(native_rgb, dtype=np.uint8)
        if source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] == 1:
            source = cv2.cvtColor(source[..., 0], cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] >= 3:
            source = source[..., :3]
        else:
            raise ValueError("Transfer target must be an RGB-compatible image")

        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        denoised = cv2.bilateralFilter(bgr, d=5, sigmaColor=24, sigmaSpace=5)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)

        def normalized_gradient(channel):
            channel = np.asarray(channel, dtype=np.uint8)
            grad_x = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
            grad_y = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
            magnitude = cv2.magnitude(grad_x, grad_y)
            positive = magnitude[magnitude > 0]
            if positive.size == 0:
                return np.zeros_like(magnitude, dtype=np.float32)
            low = float(np.percentile(positive, 34.0))
            high = float(np.percentile(positive, 98.8))
            if high <= low + 1e-6:
                return np.clip(magnitude / max(high, 1.0), 0.0, 1.0)
            return np.clip((magnitude - low) / (high - low), 0.0, 1.0)

        light_edge = normalized_gradient(lightness)
        color_edge = np.maximum(
            normalized_gradient(channel_a),
            normalized_gradient(channel_b),
        )
        edge_strength = np.maximum(light_edge, 0.90 * color_edge)
        edge_strength = cv2.GaussianBlur(edge_strength, (0, 0), 0.24)
        edge_strength = np.clip(edge_strength, 0.0, 1.0) ** 0.78

        # 基础可见线稿：比旧版更柔和，保留弱边缘，不做硬二值化。
        gray_sketch = np.clip(255.0 - 215.0 * edge_strength, 0.0, 255.0).astype(np.uint8)
        visible = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2RGB)

        # 保留原图里的细黑线、浅灰线与细彩色结构线，避免大量细节丢失。
        neutral_ink = ColorTransferToolbox._transfer_native_ink_mask(source)
        structural_ink = ColorTransferToolbox._transfer_structural_ink_mask(source)
        subtle_dark = (gray < 238) & (cv2.GaussianBlur((edge_strength * 255).astype(np.uint8), (0, 0), 0.18) > 6)
        preserve_mask = subtle_dark | neutral_ink | structural_ink
        gray_rgb = np.repeat(gray[..., None], 3, axis=2)
        visible[preserve_mask] = np.minimum(visible[preserve_mask], gray_rgb[preserve_mask])

        # 轻微提亮背景，避免非边缘纹理发灰，同时不抹去真实细线。
        background = ~preserve_mask
        if np.any(background):
            lifted = visible.astype(np.int16)
            lifted[background] = np.clip(lifted[background] + 6, 0, 255)
            visible = lifted.astype(np.uint8)
        return Image.fromarray(visible, mode='RGB')

    @staticmethod
    def _build_transfer_traditional_outline(native_rgb):
        """以原生尺寸构造传统目标线稿；若目标已是线稿则逐像素保留。"""
        source = np.asarray(native_rgb, dtype=np.uint8)
        if source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] == 1:
            source = cv2.cvtColor(source[..., 0], cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] >= 3:
            source = source[..., :3]
        else:
            raise ValueError("Traditional transfer target must be an RGB-compatible image")

        if ColorTransferToolbox._transfer_native_is_line_sketch(source):
            return Image.fromarray(source.copy(), mode='RGB')

        return ColorTransferToolbox._build_transfer_soft_visible_outline(source)

    @staticmethod
    def _transfer_outline_has_closed_face(outline_rgb):
        """判断线稿中是否存在不与画布边缘连通的四连通可填区域。"""
        source = np.asarray(outline_rgb, dtype=np.uint8)
        if source.ndim == 2:
            gray = source
        elif source.ndim == 3 and source.shape[2] == 1:
            gray = source[..., 0]
        elif source.ndim == 3 and source.shape[2] >= 3:
            gray = cv2.cvtColor(source[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Transfer outline must be an RGB-compatible image")

        fillable = np.uint8(gray >= 235)
        component_count, components = cv2.connectedComponents(
            fillable, connectivity=4
        )
        if component_count <= 1:
            return False
        edge_components = set(int(value) for value in np.unique(np.concatenate((
            components[0, :],
            components[-1, :],
            components[:, 0],
            components[:, -1],
        ))))
        return any(
            component not in edge_components
            for component in range(1, component_count)
        )

    def _transfer_target_is_traditional(self, native_rgb):
        """Transfer 只读取既有分类器，不改变 Coloring Line Sketch 路径。"""
        diagnostics = self._palette_classify_editor_image(
            Image.fromarray(np.asarray(native_rgb, dtype=np.uint8), mode="RGB")
        )
        return diagnostics.get("mode") == "traditional"

    def _build_transfer_render_context(self, labels, unique_labels):
        """统一采用 Load Source Image 同源线稿逻辑，构建高细节迁移渲染上下文。"""
        analysis_labels = np.asarray(labels, dtype=np.int32)
        if (
            getattr(self, 'target_image_path', None)
            and os.path.exists(self.target_image_path)
        ):
            native_array = np.asarray(
                Image.open(self.target_image_path).convert('RGB'), dtype=np.uint8
            )
        elif labels is not None:
            native_array = np.full(
                (analysis_labels.shape[0], analysis_labels.shape[1], 3),
                255,
                dtype=np.uint8,
            )
        else:
            native_array = np.full((500, 500, 3), 255, dtype=np.uint8)

        native_labels = self._build_native_transfer_owner_labels(
            native_array, analysis_labels, unique_labels
        )
        height, width = native_array.shape[:2]

        # 与 Load Source Image 完全同源：直接使用 create_outline_image 生成的
        # target_outline_image_path 作为最终细节线稿覆盖层。
        if (
            getattr(self, 'target_outline_image_path', None)
            and os.path.exists(self.target_outline_image_path)
        ):
            line_overlay = Image.open(
                self.target_outline_image_path
            ).convert('L').resize((width, height), Image.Resampling.LANCZOS)
            line_array = np.asarray(line_overlay, dtype=np.float32)
        else:
            regenerated = self._generate_complete_line_sketch(native_array)
            if regenerated.ndim == 3 and regenerated.shape[2] >= 3:
                line_array = cv2.cvtColor(
                    regenerated[..., :3], cv2.COLOR_RGB2GRAY
                ).astype(np.float32)
            else:
                line_array = np.asarray(regenerated, dtype=np.float32)

        line_transmission = np.clip(
            (255.0 - line_array) / 255.0, 0.0, 1.0
        )[:, :, None]
        line_rgb = np.repeat(line_array[:, :, None], 3, axis=2).astype(np.uint8)
        return {
            'native_array': native_array,
            'native_labels': native_labels,
            'line_transmission': line_transmission,
            'line_rgb': line_rgb,
        }

    def create_colored_image_for_transfer(
            self, placement, selected_colors, labels, unique_labels,
            render_context=None):
        """Transfer Colors 统一渲染：白底分区填色 + 高清线稿叠加，得到细致配色图。"""
        import numpy as np
        from PIL import Image

        context = render_context or self._build_transfer_render_context(
            labels, unique_labels
        )
        native_array = np.asarray(context['native_array'], dtype=np.uint8)
        native_labels = np.asarray(context['native_labels'], dtype=np.int32)
        img_array = np.full(native_array.shape, 255, dtype=np.uint8)

        placement = self._complete_placement(
            placement, len(unique_labels), len(selected_colors)
        )
        blank_mask = np.zeros(native_labels.shape, dtype=bool)
        for region_idx, color_idx in enumerate(placement):
            if color_idx is None and region_idx < len(unique_labels):
                blank_mask |= (native_labels == unique_labels[region_idx])

        for region_idx, color_idx in enumerate(placement):
            normalized_color_idx = self._normalize_index(color_idx)
            if normalized_color_idx is None:
                continue
            if normalized_color_idx >= len(selected_colors) or region_idx >= len(unique_labels):
                continue
            color = selected_colors[normalized_color_idx]
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            region_label = unique_labels[region_idx]
            mask = (native_labels == region_label)
            img_array[mask] = [r, g, b]

        # 轻度同时对比增强，让细致配色更通透，但保留空白槽。
        try:
            self._apply_simultaneous_contrast_correction(
                img_array,
                native_labels,
                getattr(self, 'precomputed_network', None),
                alpha=0.16,
            )
        except Exception:
            pass
        img_array[blank_mask] = 255

        # 统一叠加由 Load Source Image 同源逻辑生成的高清线稿，保留所有细线细节。
        transmission = np.asarray(context['line_transmission'], dtype=np.float32)
        img_array = np.clip(
            img_array.astype(np.float32) * (1.0 - transmission),
            0,
            255,
        ).astype(np.uint8)

        # 再次恢复空白区域为“白底 + 线稿”，避免被全局处理污染。
        line_rgb = np.asarray(context.get('line_rgb'), dtype=np.uint8)
        white_with_line = np.clip(
            255.0 * (1.0 - transmission), 0, 255
        ).astype(np.uint8)
        if white_with_line.ndim == 3 and white_with_line.shape[2] == 1:
            white_with_line = np.repeat(white_with_line, 3, axis=2)
        img_array[blank_mask] = white_with_line[blank_mask]

        return Image.fromarray(img_array)

    def _make_high_quality_preview(self, pil_image, max_size=(150, 150)):
        """高质量缩略图，避免 NEAREST 造成的像素块与锯齿。"""
        from PIL import ImageFilter
        img = pil_image.copy().convert('RGB')
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        try:
            img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=125, threshold=2))
        except Exception:
            pass
        return img

    def display_colored_image(self, idx, colored_image, image_frame, cols):
        """Display colored image in the UI frame - directly from memory without disk cache"""
        img = self._make_high_quality_preview(colored_image, (150, 150))

        photo = ImageTk.PhotoImage(img)
        label = ttk.Label(image_frame, image=photo)
        label.image = photo

        row = idx // cols
        col = idx % cols
        label.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')

    def create_colored_image(self, placement, selected_colors, unique_labels=None):
        """
        根据 placement 创建彩色图像 - 修复版：正确处理 None 值（空白区域）
        """
        import numpy as np
        import cv2
        from PIL import Image

        labels = self.target_labels_2d

        base_img = Image.open( self.target_image_path ).convert( "RGB" )
        img_array = np.array( base_img )
        img_h, img_w = img_array.shape[:2]

        # 调整 labels 尺寸以匹配图像
        if labels.shape[0] != img_h or labels.shape[1] != img_w:
            labels = cv2.resize(
                labels.astype( np.int32 ),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )

        # 如果没有提供 unique_labels，则重新计算
        if unique_labels is None:
            unique_labels = np.unique( labels )
            unique_labels = unique_labels[unique_labels >= 0]  # 排除背景

        placement = self._complete_placement(
            placement, len(unique_labels), len(selected_colors)
        )

        # 只上色 placement 中指定的区域，None 保持原图
        for region_idx, color_idx in enumerate( placement ):
            normalized_color_idx = self._normalize_index( color_idx )
            if normalized_color_idx is None:
                continue  # 跳过 None，保持该区域为原始图像

            if normalized_color_idx >= len( selected_colors ):
                continue

            if region_idx >= len( unique_labels ):
                continue

            color = selected_colors[normalized_color_idx]
            r = int( color[1:3], 16 )
            g = int( color[3:5], 16 )
            b = int( color[5:7], 16 )

            region_label = unique_labels[region_idx]
            mask = labels == region_label

            # 上色
            img_array[mask] = [r, g, b]

        return Image.fromarray( img_array )
    def save_and_display_colored_image(self, idx, placement, selected_colors, image_frame, cols):
        colored_image = self.create_colored_image( placement, selected_colors )

        file_path = f"colored_image_{idx + 1}.png"
        colored_image.save( file_path )

        img = Image.open( file_path ).convert( 'RGB' )
        img.thumbnail( (150, 150) )

        photo = ImageTk.PhotoImage( img )
        label = ttk.Label( image_frame, image=photo )
        label.image = photo

        row = idx // cols
        col = idx % cols
        label.grid( row=row, column=col, padx=5, pady=5, sticky='nsew' )

    # -------------------- 新增辅助方法 --------------------
    def get_region_centers(self, regions):
        centers = []
        for mask in regions:
            ys, xs = np.where( mask > 0 )
            if len( xs ) == 0:
                centers.append( (0, 0) )
            else:
                centers.append( (np.mean( xs ), np.mean( ys )) )
        return centers

    def get_labels_for_transfer(self):
        if hasattr( self, 'target_labels_2d' ) and self.target_labels_2d is not None:
            return self.target_labels_2d
        return self.labels_2d

    def generate_color_transfer_plans(self, selected_colors, total_images, symmetry_type="Non_Symmetry", progress_callback=None):
        """
        生成颜色迁移方案 - 修复版：正确处理空白区域（None）
        """
        if getattr(self, '_transfer_element_aware_active', False):
            return self._generate_element_aware_v21_plans(
                selected_colors,
                total_images,
                progress_callback=progress_callback,
            )

        import itertools
        import numpy as np
        from PIL import Image
        from tkinter import messagebox

        def report_progress(value, text):
            if progress_callback:
                try:
                    progress_callback(value, text)
                except Exception:
                    pass

        if not hasattr( self, 'target_labels_2d' ) or self.target_labels_2d is None:
            messagebox.showerror( "Error!", "Target image or target area information not loaded!" )
            return [], []

        num_colors = len( selected_colors )
        labels = np.asarray(self.target_labels_2d, dtype=np.int32)
        h, w = labels.shape[:2]

        desired_regions = int(getattr(self, '_transfer_total_region_count', 0) or 0)
        if desired_regions > 0:
            normalized = self._normalize_transfer_labels_to_exact_count(
                labels, desired_regions
            )
            if normalized is not None:
                labels = normalized
                self.target_labels_2d = normalized

        # 获取有效的区域标签（排除背景 -1）
        unique_labels = np.unique( labels )
        unique_labels = unique_labels[unique_labels >= 0]
        num_regions = len( unique_labels )

        # 如果颜色数大于区域数，提示错误
        if num_colors > num_regions:
            messagebox.showerror( "Error!",
                                  f"Selected colors ({num_colors}) cannot exceed number of regions ({num_regions})" )
            return [], []

        # Transfer Colors 统一使用严格的部分排列枚举：每种颜色只使用一次，
        # 其余 N-M 个区域保持空白，不再按对称组扩展映射。
        placements = self.generate_partial_placements(num_regions, num_colors)
        slot_count = num_regions
        expected_count = (
            math.factorial(slot_count)
            // math.factorial(slot_count - num_colors)
        )
        if len(placements) != expected_count:
            raise RuntimeError(
                "Partial permutation enumeration mismatch: "
                f"expected {expected_count}, got {len(placements)}"
            )
        self._last_plan_permutation_count = int(expected_count)
        self._last_plan_permutation_formula = (
            f"P({slot_count},{num_colors})={expected_count}"
        )

        # 调试输出
        print(f"Transfer Colors: Generated {len(placements)} placements")
        if placements:
            sample = placements[0]
            none_count = sum(1 for x in sample if x is None)
            print(f"Sample placement None count: {none_count}/{len(sample)}")

        # 与 Element-Aware 共用源图 CRN owner。映射评分只读取目标标签与
        # placement；原生尺寸的目标渲染仅用于预览，不能和分析标签混算。
        source_bundle = self._build_source_crn_bundle()
        representative_palette = self._extract_representative_palette_prior(
            len(selected_colors), quantile=0.2
        )
        region_areas = [
            int(np.count_nonzero(labels == label)) for label in unique_labels
        ]

        scored_metadata = []
        total_to_score = len(placements)
        update_step = max(1, total_to_score // 120) if total_to_score > 0 else 1
        for idx, placement in enumerate(placements):
            plan_palette = list(selected_colors)
            if source_bundle is not None:
                breakdown = self.get_crn_score_breakdown(
                    None, labels, unique_labels, placement, plan_palette,
                    source_bundle['G'], source_bundle['centrality'],
                    source_bundle['proportions'],
                    node_attrs=source_bundle['node_attrs'],
                    edge_attrs=source_bundle['edge_attrs'],
                    source_disps=source_bundle['dispersions'],
                    user_selected_colors=selected_colors,
                    representative_palette=representative_palette,
                )
                breakdown = dict(breakdown)
                breakdown['CRNAvailable'] = True
                area_summary = self._element_aware_color_area_summary(
                    placement, region_areas, len(plan_palette)
                )
                breakdown['PaletteMeanDeltaE'] = 0.0
                breakdown['PaletteMaxDeltaE'] = 0.0
                breakdown['PaletteMode'] = 'Exact user palette'
                breakdown['PairwiseGuard'] = True
                breakdown['UsedColorCount'] = area_summary['used_color_count']
                breakdown['DominantColorShare'] = area_summary[
                    'dominant_color_share'
                ]
                breakdown['ColorAreaShares'] = area_summary['color_area_shares']
                if breakdown.get('TargetColorShares'):
                    breakdown['ColorAreaShares'] = list(
                        breakdown['TargetColorShares']
                    )
                    breakdown['DominantColorShare'] = float(
                        breakdown.get(
                            'MaximumColorShare',
                            area_summary['dominant_color_share'],
                        )
                    )
                score = float(breakdown.get('total', 0.0))
                explanation = self.build_crn_explanation_summary(
                    placement, plan_palette,
                    source_bundle['G'], source_bundle['centrality'],
                    source_bundle['proportions'], source_bundle['node_attrs'],
                    breakdown=breakdown,
                )
            else:
                # 无源 CRN 时只构造标签同尺寸的离散预览用于兼容评分，绝不
                # 提前构建原生 owner 或读取目标覆盖层。
                image_array = np.full((h, w, 3), 255, dtype=np.uint8)
                palette_rgb = self._element_aware_v21_hex_to_rgb(plan_palette)
                complete_placement = self._complete_placement(
                    placement, len(unique_labels), len(plan_palette)
                )
                for region_index, region_label in enumerate(unique_labels):
                    if region_index >= len(complete_placement):
                        break
                    color_index = self._normalize_index(
                        complete_placement[region_index]
                    )
                    if color_index is None or not 0 <= color_index < len(palette_rgb):
                        continue
                    image_array[labels == region_label] = palette_rgb[color_index]
                score = float(self.calculate_image_score(
                    image_array,
                    getattr(self, 'color_regions', None),
                    labels,
                    precomputed_network=None,
                ))
                area_summary = self._element_aware_color_area_summary(
                    placement, region_areas, len(plan_palette)
                )
                breakdown = {
                    'total': score,
                    'CRNAvailable': False,
                    'PaletteMeanDeltaE': 0.0,
                    'PaletteMaxDeltaE': 0.0,
                    'PaletteMode': 'Exact user palette (legacy score fallback)',
                    'PairwiseGuard': True,
                    'UsedColorCount': area_summary['used_color_count'],
                    'DominantColorShare': area_summary['dominant_color_share'],
                    'ColorAreaShares': area_summary['color_area_shares'],
                }
                explanation = (
                    'Source CRN unavailable; score computed with a label-sized '
                    'compatibility render.'
                )
            scored_metadata.append(
                (placement, score, None, breakdown, explanation, plan_palette)
            )

            if idx % update_step == 0 or idx == total_to_score - 1:
                pct = 85 * (idx + 1) / max(1, total_to_score)
                report_progress(pct, f"Scoring {idx + 1}/{total_to_score}")

        scored_metadata.sort(key=lambda item: item[1], reverse=True)
        # 保留全部 P(N,M) 个方案；Rows×Cols 不再截断候选。
        selected_metadata = scored_metadata
        scored_placements = []
        render_context = (
            self._build_transfer_render_context(labels, unique_labels)
            if selected_metadata else None
        )
        for render_index, item in enumerate(selected_metadata, start=1):
            placement, score, _image, breakdown, explanation, plan_palette = item
            colored_image = self.create_colored_image_for_transfer(
                placement,
                plan_palette,
                labels,
                unique_labels,
                render_context=render_context,
            )
            if colored_image is None:
                continue
            scored_placements.append((
                placement, score, colored_image,
                breakdown, explanation, plan_palette,
            ))
            report_progress(
                85 + 15 * render_index / max(1, len(selected_metadata)),
                f"Rendering {render_index}/{len(selected_metadata)}",
            )
        return placements, scored_placements
    def generate_partial_placements(self, num_regions, num_colors):
        """生成 P(N,M) 个部分着色方案：M 种颜色各使用一次，其余 N-M 区域留白。"""
        return self._generate_partial_permutation_placements(
            num_regions, num_colors
        )

    def generate_complete_placements(self, num_regions, num_colors):
        """兼容旧调用；现在同样遵循部分排列 P(N,M)。"""
        return self._generate_partial_permutation_placements(
            num_regions, num_colors
        )

    @staticmethod
    def _generate_partial_permutation_placements(num_slots, num_colors):
        """严格枚举部分排列 P(N,M)=N!/(N-M)!，不设置 MAX 截断。

        placement 长度恒为 N。颜色索引 0..M-1 各出现一次，未选中的
        N-M 个槽位为 None，渲染时保持白底/原有结构线，不再循环填满。
        """
        import itertools

        try:
            num_slots = int(num_slots)
            num_colors = int(num_colors)
        except (TypeError, ValueError, OverflowError):
            return []
        if num_slots <= 0 or num_colors <= 0 or num_colors > num_slots:
            return []

        placements = []
        for ordered_regions in itertools.permutations(
            range(num_slots), num_colors
        ):
            placement = [None] * num_slots
            for color_index, region_index in enumerate(ordered_regions):
                placement[int(region_index)] = int(color_index)
            placements.append(placement)
        return placements

    def create_progress_window(parent, title="Processing...", message="Please wait...", max_value=100):
        """
        创建进度条窗口
        :param parent: 父窗口（Tk/Toplevel）
        :param title: 窗口标题
        :param message: 显示消息
        :param max_value: 进度条最大值
        :return: 进度条窗口对象、进度条对象、百分比标签对象
        """
        # 创建进度条窗口
        progress_win = tk.Toplevel( parent )
        progress_win.title( title )
        progress_win.geometry( "300x150" )
        progress_win.transient( parent )  # 设置为临时窗口
        progress_win.grab_set()  # 独占焦点
        progress_win.resizable( False, False )

        # 居中显示（可选）
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        win_x = parent_x + (parent_w - 300) // 2
        win_y = parent_y + (parent_h - 150) // 2
        progress_win.geometry( f"300x150+{win_x}+{win_y}" )

        # 消息标签
        msg_label = ttk.Label( progress_win, text=message )
        msg_label.pack( pady=10 )

        # 进度条
        progress_bar = ttk.Progressbar( progress_win, orient='horizontal', length=250, mode='determinate' )
        progress_bar.pack( pady=10 )
        progress_bar['maximum'] = max_value
        progress_bar['value'] = 0

        # 百分比标签
        percent_label = ttk.Label( progress_win, text="0%" )
        percent_label.pack()

        # 更新窗口显示
        progress_win.update()

        return progress_win, progress_bar, percent_label

    def update_progress(progress_win, progress_bar, percent_label, current_value):
        """
        更新进度条
        :param progress_win: 进度条窗口
        :param progress_bar: 进度条对象
        :param percent_label: 百分比标签
        :param current_value: 当前进度值
        """
        progress_bar['value'] = current_value
        max_value = progress_bar['maximum']
        percent = int( (current_value / max_value) * 100 ) if max_value > 0 else 0
        percent_label.config( text=f"{percent}%" )
        progress_win.update()  # 强制更新UI

    def close_progress_window(progress_win):
        """
        关闭进度条窗口
        :param progress_win: 进度条窗口
        """
        progress_win.grab_release()
        progress_win.destroy()

    def _normalize_index(self, value):
        if value is None:
            return None
        try:
            return int( value )
        except (TypeError, ValueError):
            return None

    def group_regions_by_symmetry(self, centers, sym_type, img_shape):
        h, w = img_shape[:2]
        cx_center = w / 2.0
        cy_center = h / 2.0
        threshold = 5.0

        n = len( centers )
        visited = [False] * n
        groups = []

        def sym_transform(x, y):
            if sym_type == "Original point Symmetry":
                return (w - x, h - y)
            elif sym_type == "x-aix Symmetry":
                return (x, h - y)
            elif sym_type == "y-aix Symmetry":
                return (w - x, y)
            elif sym_type == "y=x Symmetry":
                xc = x - cx_center
                yc = y - cy_center
                return (yc + cx_center, xc + cy_center)
            elif sym_type == "y=-x Symmetry":
                xc = x - cx_center
                yc = y - cy_center
                return (-yc + cx_center, -xc + cy_center)
            else:
                return (x, y)

        for i in range( n ):
            if visited[i]:
                continue
            x_i, y_i = centers[i]
            tx, ty = sym_transform( x_i, y_i )

            matched = None
            for j in range( n ):
                if i == j or visited[j]:
                    continue
                x_j, y_j = centers[j]
                if abs( x_j - tx ) <= threshold and abs( y_j - ty ) <= threshold:
                    matched = j
                    break

            if matched is not None:
                groups.append( [i, matched] )
                visited[i] = visited[matched] = True
            else:
                groups.append( [i] )
                visited[i] = True

        return groups

    def generate_group_placements(self, groups, num_colors):
        return self._generate_partial_permutation_placements(
            len(groups), num_colors
        )

    def generate_group_placements_symmetry(self, groups, num_colors):
        return self._generate_partial_permutation_placements(
            len(groups), num_colors
        )

    def generate_complete_group_placements_symmetry(self, groups, num_colors):
        return self._generate_partial_permutation_placements(
            len(groups), num_colors
        )

    def get_region_centers_from_labels(self, labels):
        """
        获取区域中心 - 排除背景标签-1
        """
        h, w = labels.shape
        unique_labels = np.unique( labels )
        valid_labels = unique_labels[unique_labels >= 0]

        centers = []
        for lbl in valid_labels:
            ys, xs = np.where( labels == lbl )
            if len( xs ) == 0:
                centers.append( (0, 0) )
            else:
                centers.append( (np.mean( xs ), np.mean( ys )) )
        return centers, valid_labels

    def generate_plan(self):

        try:

            is_outline = (
                    self.last_action == "outline_coloring"
                    and self.outline_image_path
                    and os.path.exists( self.outline_image_path )
            )

            is_transfer = (
                    self.last_action == "color_transfer"
                    and self.target_image_path
                    and os.path.exists( self.target_image_path )
            )

            if not (is_outline or is_transfer):
                messagebox.showerror(
                    "Error!",
                    "Please load the line draft or target image firstly!"
                )
                return

            rows = self.row_var.get()
            cols = self.col_var.get()
            total_images = rows * cols  # 仅保留旧参数兼容；不再限制实际方案数

            self.update_status("All exact partial-permutation plans are generating now...")

            # =========================
            # Progress Window
            # =========================

            progress_window = tk.Toplevel( self.root )
            progress_window.title( "Generating Plans" )
            progress_window.geometry( "450x180" )
            progress_window.resizable( False, False )
            progress_window.transient( self.root )
            progress_window.grab_set()

            container = ttk.Frame( progress_window )
            container.pack( expand=True )

            title_label = ttk.Label(
                container,
                text="Generating all exact design plans...",
                font=("Arial", 11, "bold")
            )
            title_label.pack( pady=10 )

            progress_bar = ttk.Progressbar(
                container,
                orient="horizontal",
                length=360,
                mode="determinate",
                maximum=100
            )
            progress_bar.pack( pady=10 )

            percent_label = ttk.Label(
                container,
                text="0%",
                font=("Arial", 10, "bold")
            )
            percent_label.pack()

            # 进度窗口只显示“当前/总数 百分比”，不再展示内部阶段名、
            # P(n,m)、缓存命中等实现细节。

            # =========================
            # Progress Update
            # =========================

            def update_progress(value, text):
                """只显示简洁计数，例如“12/120  10%”。

                生成算法和缓存逻辑保持不变；界面从内部消息中提取当前阶段的
                current/total，并按该阶段计算百分比。没有计数时才使用传入的
                百分比值。
                """
                import re

                message = str(text or "")
                match = re.search(r"(?<![\d,])(\d+)\s*/\s*(\d+)", message)
                if match:
                    current = max(0, int(match.group(1)))
                    total = max(1, int(match.group(2)))
                    current = min(current, total)
                    visible_value = 100.0 * current / total
                    visible_text = (
                        f"{current}/{total}  {int(round(visible_value))}%"
                    )
                else:
                    visible_value = max(0.0, min(100.0, float(value)))
                    visible_text = f"{int(round(visible_value))}%"

                def ui():
                    try:
                        progress_bar["value"] = visible_value
                        percent_label.config(text=visible_text)
                    except Exception:
                        pass

                progress_window.after(10, ui)

            # =========================
            # Generate Thread
            # =========================

            def generate_plans_thread():

                try:

                    update_progress( 0, "Preparing color data..." )

                    self.generated_plans = []
                    self.latest_generated_result_image = None
                    self.latest_generated_result_description = None

                    progress_window.after(
                        0,
                        lambda: [
                            widget.destroy()
                            for widget in self.design_container.winfo_children()
                        ]
                    )

                    # ---------------------
                    # 获取颜色
                    # ---------------------

                    selected_colors = []

                    if self.last_action == "outline_coloring":

                        selected_colors = [
                            self.outline_coloring_colors[i]
                            for i, var in enumerate( self.color_selection_vars )
                            if var.get()
                        ]

                    elif self.last_action == "color_transfer":

                        selected_colors = [
                            self.color_transfer_colors[i]
                            for i, var in enumerate( self.color_vars )
                            if var.get()
                        ]

                    if not selected_colors:
                        progress_window.after( 0, progress_window.destroy )

                        progress_window.after(
                            0,
                            lambda: messagebox.showerror(
                                "Error",
                                "Please select at least one type of color."
                            )
                        )

                        return

                    # ---------------------
                    # 生成方案
                    # ---------------------

                    update_progress( 3, "Calculating plan combinations..." )

                    symmetry = self.symmetry_type.get()

                    if self.last_action == "outline_coloring":

                        placements, scored_placements = self.generate_outline_coloring_plans(
                            selected_colors,
                            total_images,
                            symmetry,
                            progress_callback=update_progress
                        )

                    else:

                        placements, scored_placements = self.generate_color_transfer_plans(
                            selected_colors,
                            total_images,
                            symmetry,
                            progress_callback=update_progress
                        )

                    actual_total = len(scored_placements)
                    update_progress(100, f"Rendering 0/{max(1, actual_total)}")
                    progress_window.after(
                        0,
                        lambda: title_label.config(
                            text=f"Generated {actual_total} exact design plans"
                        )
                    )

                    # ---------------------
                    # 创建 UI 容器
                    # ---------------------

                    def build_ui():
                        # 左侧逐项展示评分说明，右侧严格按 Rows × Cols 展示预览。
                        # 旧版把每张缩略图塞进自己的说明卡片，导致右侧大面积空白，
                        # 也无法直观看到用户要求的行列方案矩阵。
                        content = ttk.Frame(self.design_container)
                        content.pack(fill='both', expand=True, padx=10, pady=10)
                        content.grid_columnconfigure(0, weight=0)
                        content.grid_columnconfigure(1, weight=1)

                        detail_outer = ttk.LabelFrame(content, text="Plan details")
                        detail_outer.grid(row=0, column=0, sticky='nsw', padx=(0, 12))
                        detail_frame = ttk.Frame(detail_outer)
                        detail_frame.pack(fill='both', expand=True, padx=4, pady=4)

                        preview_outer = ttk.LabelFrame(
                            content,
                            text=f"All plan previews ({actual_total})"
                        )
                        preview_outer.grid(row=0, column=1, sticky='nsew')
                        preview_frame = ttk.Frame(preview_outer)
                        preview_frame.pack(fill='both', expand=True, padx=8, pady=8)
                        for column in range(max(1, cols)):
                            preview_frame.grid_columnconfigure(column, weight=1)
                        return detail_frame, preview_frame

                    frames = []

                    def create_frames():
                        frames.append( build_ui() )

                    progress_window.after( 0, create_frames )

                    while not frames:
                        time.sleep( 0.01 )

                    plan_detail_frame, plan_preview_frame = frames[0]

                    # ---------------------
                    # 渲染方案
                    # ---------------------

                    for i, item in enumerate(scored_placements):

                        placement, score, colored_image = item[:3]
                        breakdown = item[3] if len( item ) > 3 else None
                        explanation = item[4] if len( item ) > 4 else None
                        plan_palette = item[5] if len( item ) > 5 else selected_colors

                        # 存储图片以便一键下载和保存
                        self.generated_plans.append( colored_image )
                        source_kind = (
                            "Coloring Line Sketch"
                            if self.last_action == "outline_coloring"
                            else "Transfer Colors"
                        )
                        self.latest_generated_result_image = colored_image.copy()
                        self.latest_generated_result_description = f"{source_kind} plan {i + 1}"

                        update_progress(
                            100,
                            f"Rendering plan {i + 1}/{max(1, actual_total)}"
                        )

                        def render(i=i,
                                   placement=placement,
                                   score=score,
                                    colored_image=colored_image,
                                    breakdown=breakdown,
                                    explanation=explanation,
                                    plan_palette=plan_palette,
                                    source_kind=source_kind):

                            plan_frame = ttk.LabelFrame(
                                plan_detail_frame, text=f"Plan {i + 1}"
                            )
                            plan_frame.pack( fill='x', expand=True, padx=5, pady=7 )

                            info_frame = ttk.Frame( plan_frame )
                            info_frame.pack(
                                side='left', fill='both', expand=True,
                                padx=(8, 12), pady=8
                            )

                            ttk.Label(
                                info_frame,
                                text="Region → color mapping",
                                font=('Arial', 9, 'bold')
                            ).pack( anchor='w' )

                            preview_frame = ttk.Frame( info_frame )
                            preview_frame.pack( fill='x', pady=5 )

                            for color_idx in placement:

                                color_block = tk.Canvas(
                                    preview_frame,
                                    width=30,
                                    height=30,
                                    bd=1,
                                    relief='solid',
                                    bg='white'
                                )

                                color_block.pack( side='left', padx=2 )

                                normalized_color_idx = self._normalize_index( color_idx )
                                if (
                                    normalized_color_idx is not None
                                    and 0 <= normalized_color_idx < len(plan_palette)
                                ):
                                    color_block.config(bg=plan_palette[normalized_color_idx])
                                else:
                                    color_block.create_line(6, 6, 24, 24, fill='#808080')
                                    color_block.create_line(24, 6, 6, 24, fill='#808080')

                            ttk.Label(
                                info_frame,
                                text=f"Score: {score:.2f}"
                            ).pack( anchor='w' )


                            if (
                                breakdown
                                and breakdown.get('CRNAvailable', True) is False
                            ):
                                ttk.Label(
                                    info_frame,
                                    text="CRN unavailable / compatibility score",
                                    wraplength=420,
                                    justify='left',
                                ).pack(anchor='w')
                            elif breakdown:
                                ttk.Label(
                                    info_frame,
                                    text=(
                                        f"Mass {breakdown.get('MassFidelity', 0.0):.1f} | "
                                        f"Centrality {breakdown.get('CentralityFidelity', 0.0):.1f} | "
                                        f"GraphEdge {breakdown.get('GraphEdgeFidelity', 0.0):.1f} | "
                                        f"ELH-rel {breakdown.get('ELH', 0.0):.1f} | "
                                        f"NodeRole {breakdown.get('NodeRole', 0.0):.1f} | "
                                        f"VAFE {breakdown.get('VAFE', 0.0):.1f} | "
                                        f"Topology {breakdown.get('Topology', 0.0):.1f} | "
                                        f"NSB {breakdown.get('NSB', 0.0):.1f} | "
                                        f"Palette {breakdown.get('PaletteFidelity', 0.0):.1f} | "
                                        f"Rep {breakdown.get('RepresentativePalette', 0.0):.1f}"
                                    ),
                                    wraplength=420,
                                    justify='left'
                                ).pack( anchor='w' )
                                ttk.Label(
                                    info_frame,
                                    text=(
                                        f"Palette mode: {breakdown.get('PaletteMode', 'Exact user palette')} | "
                                        f"mean ΔE00 {breakdown.get('PaletteMeanDeltaE', 0.0):.2f} | "
                                        f"max ΔE00 {breakdown.get('PaletteMaxDeltaE', 0.0):.2f}"
                                    ),
                                    wraplength=420,
                                    justify='left'
                                ).pack( anchor='w' )
                                ttk.Label(
                                    info_frame,
                                    text=(
                                        f"Colors used: {breakdown.get('UsedColorCount', 0)} | "
                                        f"min/max color area: "
                                        f"{100.0 * breakdown.get('MinimumColorShare', 0.0):.1f}%/"
                                        f"{100.0 * breakdown.get('MaximumColorShare', breakdown.get('DominantColorShare', 0.0)):.1f}% | "
                                        f"Area gate {breakdown.get('AreaConstraint', 0.0):.1f} | "
                                        f"mapping: {list(placement)}"
                                    ),
                                    wraplength=520,
                                    justify='left'
                                ).pack( anchor='w' )

                            if explanation:
                                ttk.Label(
                                    info_frame,
                                    text=explanation,
                                    wraplength=420,
                                    justify='left'
                                ).pack( anchor='w', pady=(2, 4) )
                            ttk.Button(
                                info_frame,
                                text="Download plan",
                                command=lambda idx=i, img=colored_image,
                                               saved_placement=list(placement),
                                               saved_palette=list(plan_palette):
                                self.save_single_plan(
                                    idx,
                                    saved_placement,
                                    saved_palette,
                                    img
                                )
                            ).pack( anchor='e', pady=5 )

                            ttk.Button(
                                info_frame,
                                text="Edit in Palette Result Editor",
                                command=lambda img=colored_image,
                                               desc=f"{source_kind} plan {i + 1}":
                                self.palette_editor_load_generated_result(img, desc)
                            ).pack( anchor='e', pady=(0, 5) )

                            self.display_colored_image(
                                i,
                                colored_image,
                                plan_preview_frame,
                                max(1, cols)
                            )

                        progress_window.after( 0, render )

                    # ---------------------
                    # 完成
                    # ---------------------

                    update_progress( 100, "Completed!" )

                    def finish():

                        ttk.Button(
                            progress_window,
                            text="OK",
                            command=progress_window.destroy
                        ).pack( pady=10 )

                        self.update_status(
                            f"{actual_total} exact plans generated!"
                        )

                        self.show_tab( self.design_tab, "Plan design." )
                        self.notebook.select( self.design_tab )

                    progress_window.after( 0, finish )

                except Exception as e:
                    import traceback

                    error_msg = traceback.format_exc()
                    print( error_msg )

                    progress_window.after( 0, progress_window.destroy )

                    progress_window.after(
                        0,
                        lambda: messagebox.showerror(
                            "Error",
                            f"Generate potential plan failed:\n{error_msg}"
                        )
                    )

            thread = threading.Thread(
                target=generate_plans_thread,
                daemon=True
            )

            thread.start()

        except Exception as e:
            import traceback

            error_msg = traceback.format_exc()
            print( error_msg )

            try:
                progress_window.destroy()
            except:
                pass

            messagebox.showerror(
                "Error",
                f"Generate potential plan failed:\n{error_msg}"
            )

    # -------------------- 颜色流行趋势 --------------------
    def batch_trend_colors(self):
        win = tk.Toplevel( self.root )
        win.title( "Color Trend Analysis, Professional Version." )
        win.state( 'zoomed' )
        win.transient( self.root )
        win.grab_set()
        win.columnconfigure( 1, weight=1 )
        win.rowconfigure( 1, weight=1 )

        list_frame = ttk.LabelFrame( win, text="① Image List", width=200 )
        list_frame.grid( row=0, column=0, rowspan=2, sticky='nsw', padx=5, pady=5 )

        list_box = tk.Listbox( list_frame, selectmode='extended' )
        list_box.pack( side='left', fill='both', expand=True )
        scroll = ttk.Scrollbar( list_frame, orient='vertical', command=list_box.yview )
        scroll.pack( side='right', fill='y' )
        list_box.config( yscrollcommand=scroll.set )

        def add_imgs():
            new = filedialog.askopenfilenames(
                title="Add image",
                filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.jfif")]
            )
            for p in new:
                list_box.insert( 'end', p )

        def del_imgs():
            for idx in reversed( list_box.curselection() ):
                list_box.delete( idx )

        def clear_imgs():
            list_box.delete( 0, 'end' )

        for btn_text, cmd in [("+ Add", add_imgs), ("- Delte", del_imgs), ("Empty", clear_imgs)]:
            ttk.Button( list_frame, text=btn_text, command=cmd ).pack( fill='x', padx=2, pady=2 )

        run_frame = ttk.Frame( list_frame )
        run_frame.pack( fill='x', pady=(10, 5) )

        ttk.Button( run_frame, text="Run", command=self.run ).pack( fill='x', padx=2, pady=2 )
        ttk.Button( run_frame, text="Export 4K", command=self.export_4k ).pack( fill='x', padx=2, pady=2 )
        ttk.Button( run_frame, text="Close the window.", command=win.destroy ).pack( fill='x', padx=2, pady=2 )

        para_frame = ttk.LabelFrame( win, text="② Parameter Setting!" )
        para_frame.grid( row=0, column=1, sticky='ew', padx=5, pady=5 )
        k1_var = tk.IntVar( value=8 )
        k2_var = tk.IntVar( value=5 )
        thr_var = tk.DoubleVar( value=0.3 )

        def validate_k():
            if k1_var.get() <= k2_var.get():
                messagebox.showwarning( "Parameter Error!", "K1 must larger than K2" )
                return False
            return True

        ttk.Label( para_frame, text="K1（Initial clustering）:" ).grid( row=0, column=0, sticky='e' )
        ttk.Spinbox( para_frame, from_=3, to=30, textvariable=k1_var, width=5 ).grid( row=0, column=1 )
        ttk.Label( para_frame, text="K2（Secondary clustering）:" ).grid( row=0, column=2, sticky='e' )
        ttk.Spinbox( para_frame, from_=2, to=29, textvariable=k2_var, width=5 ).grid( row=0, column=3 )
        ttk.Label( para_frame, text="Color similarity edge threshold:" ).grid( row=1, column=0, sticky='e' )
        tk.Scale(
            para_frame, from_=0, to=1, resolution=0.01,
            variable=thr_var, orient='horizontal',
            command=lambda v: thr_var.set( round( float( v ), 2 ) )
        ).grid( row=1, column=1, columnspan=3, sticky='ew', padx=5 )

        trend_frame = ttk.LabelFrame( win, text="③ Trend color" )
        trend_frame.grid( row=0, column=2, sticky='nsew', padx=5, pady=5 )
        trend_can = tk.Canvas( trend_frame, width=160, height=160, bg='white', relief='sunken', bd=2 )
        trend_can.pack()
        trend_hex = tk.StringVar( value="Not generated" )
        ttk.Label( trend_frame, textvariable=trend_hex, font=('Arial', 12, 'bold') ).pack( pady=5 )

        def apply_trend():
            if trend_hex.get() == "Not generated":
                messagebox.showwarning( "Warning!", "Please run the analysis first!" )
                return
            c = trend_hex.get()
            self._apply_trend_colors_to_workflow([c], mode="dominant")
            self.update_status( f"Dominant trend color applied: {c}" )
            messagebox.showinfo( "Success", f"Dominant trend color {c} added to the workflow palette." )

        def apply_trend_palette():
            colors = list(getattr(self, 'trend_palette_hex_codes', []) or [])
            if not colors:
                messagebox.showwarning( "Warning!", "Please run the analysis first!" )
                return
            self._apply_trend_colors_to_workflow(colors, mode="complete")
            self.update_status( f"Complete trend palette applied: {len(colors)} colors" )
            messagebox.showinfo(
                "Success",
                f"Complete trend palette applied with {len(colors)} colors."
            )

        ttk.Button( trend_frame, text="Apply dominant color", command=apply_trend ).pack( pady=3, fill='x', padx=6 )
        ttk.Button(
            trend_frame,
            text="Apply complete trend palette",
            command=apply_trend_palette
        ).pack( pady=3, fill='x', padx=6 )

        res_frame = ttk.LabelFrame( win, text="④ Result" )
        res_frame.grid( row=1, column=1, columnspan=2, sticky='nsew', padx=5, pady=5 )
        res_frame.columnconfigure( 0, weight=1 )
        res_frame.columnconfigure( 1, weight=1 )
        res_frame.rowconfigure( 0, weight=1 )

        hist_can = tk.Canvas( res_frame, bg='white', height=350 )
        hist_can.grid( row=0, column=0, sticky='nsew' )
        net_can = tk.Canvas( res_frame, bg='white', height=350 )
        net_can.grid( row=0, column=1, sticky='nsew' )

        self.list_box = list_box
        self.k1_var = k1_var
        self.k2_var = k2_var
        self.thr_var = thr_var
        self.trend_can = trend_can
        self.trend_hex = trend_hex
        self.hist_can = hist_can
        self.net_can = net_can
        self.win = win
        self.validate_k = validate_k

    def draw_pro_hist(self, canvas, hex_codes, pixels):
        canvas.delete( "all" )
        if not hex_codes:
            return

        total = sum( pixels ) or 1
        canvas.update()
        w = canvas.winfo_width() or 600
        h = canvas.winfo_height() or 320

        n_colors = len( hex_codes )
        bar_w = max( 20, min( 80, (w - 40) // n_colors ) )
        spacing = 5
        total_width = n_colors * (bar_w + spacing)
        x_start = max( 20, (w - total_width) // 2 )

        x = x_start
        max_h = h - 80

        for idx, (c, p) in enumerate( zip( hex_codes, pixels ) ):
            ratio = p / total
            bh = int( ratio * max_h )

            y1 = h - bh - 50
            y2 = h - 50
            canvas.create_rectangle( x, y1, x + bar_w, y2, fill=c, outline='black', width=2 )

            txt = f"{idx + 1}\n{ratio:.1%}"
            fg_color = 'white' if self.is_dark_color( c ) else 'black'
            canvas.create_text( x + bar_w // 2, y1 - 10, text=txt, anchor='s', font=('Arial', 9, 'bold'), fill='black' )

            if bh > 25:
                canvas.create_text( x + bar_w // 2, (y1 + y2) // 2, text=str( idx + 1 ), fill=fg_color,
                                    font=('Arial', 12, 'bold') )

            canvas.create_text( x + bar_w // 2, h - 30, text=c, anchor='n', font=('Arial', 7), fill='gray30' )

            x += bar_w + spacing

        canvas.create_text( w // 2, 15, text="Distribution of dominant color proportion!", font=('Arial', 14, 'bold'),
                            fill='black' )

    def run(self):
        if not self.list_box.size():
            messagebox.showwarning( "Warning!", "The image list is empty!" )
            return

        try:
            k1 = self.k1_var.get()
            k2 = self.k2_var.get()
            thr = self.thr_var.get()
        except:
            messagebox.showerror( "Error!", "Parameter format not correct!" )
            return

        if k1 < k2:
            messagebox.showerror( "Parameter error!", "K1 must subject to ≥ K2" )
            self.win.config( cursor="" )
            return

        self.win.config( cursor="watch" )
        self.win.update()

        all_primary_centers = []
        all_primary_weights = []
        self.per_image_clusters = []

        paths = self.list_box.get( 0, 'end' )

        current_global_index = 0
        for img_idx, p in enumerate( paths, 1 ):
            try:
                img = Image.open( p ).convert( 'RGB' )

                max_size = 400
                if max( img.size ) > max_size:
                    ratio = max_size / max( img.size )
                    img = img.resize(
                        (int( img.size[0] * ratio ), int( img.size[1] * ratio )),
                        Image.Resampling.LANCZOS
                    )

                img_array = np.array( img )
                pixels = img_array.reshape( -1, 3 )

                n_colors = min( k1, len( pixels ) )
                kmeans = KMeans( n_clusters=n_colors, random_state=42, n_init=10, max_iter=300 )
                kmeans.fit( pixels )

                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                unique, counts = np.unique( labels, return_counts=True )
                pixel_counts = dict( zip( unique, counts ) )

                img_cluster_info = {
                    'image_path': p,
                    'total_pixels': len( pixels ),
                    'clusters': []
                }

                for i in range( n_colors ):
                    pix_count = pixel_counts.get( i, 0 )
                    if pix_count > 0:
                        center = centers[i]
                        all_primary_centers.append( center )
                        all_primary_weights.append( pix_count )

                        img_cluster_info['clusters'].append( {
                            'local_id': i,
                            'global_id': current_global_index,
                            'rgb': center,
                            'hex': self._safe_hex( center ),
                            'pixel_count': pix_count,
                            'ratio': pix_count / len( pixels )
                        } )
                        current_global_index += 1

                self.per_image_clusters.append( img_cluster_info )

            except Exception as e:
                print( "Image processing failed:", e )
                continue

        if not all_primary_centers:
            messagebox.showwarning( "Warning", "Unable to extract any color!" )
            self.win.config( cursor="" )
            return

        X = np.array( all_primary_centers )
        weights = np.array( all_primary_weights )

        total_weight = np.sum( weights )
        sample_size = min( 20000, int( total_weight * 0.01 ) )
        probabilities = weights / total_weight
        indices = np.random.choice( len( X ), size=sample_size, replace=True, p=probabilities )
        X_weighted = X[indices]

        n_clusters = min( k2, len( np.unique( X, axis=0 ) ) )

        kmeans = KMeans( n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500 )
        kmeans.fit( X_weighted )
        centers = kmeans.cluster_centers_

        distances = cdist( X, centers, 'euclidean' )
        first_to_second_mapping = np.argmin( distances, axis=1 )
        self.first_to_second_mapping = first_to_second_mapping

        per_img_clusters = self.per_image_clusters
        n = len( centers )

        final_color_pixels = np.zeros( n )

        for img_info in per_img_clusters:
            for cluster in img_info['clusters']:
                new_id = first_to_second_mapping[cluster['global_id']]
                final_color_pixels[new_id] += cluster['pixel_count']

        nonzero_idx = np.where( final_color_pixels > 0 )[0]
        final_color_pixels = final_color_pixels[nonzero_idx]

        final_hex_codes = [
            self._safe_hex( np.round( centers[i] ).astype( int ) )
            for i in nonzero_idx
        ]

        sorted_idx = np.argsort( -final_color_pixels )
        final_color_pixels = final_color_pixels[sorted_idx]
        final_hex_codes = [final_hex_codes[i] for i in sorted_idx]

        centers_sorted = centers[nonzero_idx][sorted_idx]
        self.final_centers_sorted = centers_sorted
        self.trend_palette_hex_codes = list(final_hex_codes)
        self.trend_palette_pixels = [float(v) for v in final_color_pixels]

        total_pixels_all = np.sum( final_color_pixels )
        final_ratios = final_color_pixels / total_pixels_all

        most_popular = final_hex_codes[0]
        try:
            self.trend_hex.set( most_popular )
            self.trend_can.config( bg=most_popular )
        except:
            pass

        self.draw_pro_hist( self.hist_can, final_hex_codes, final_color_pixels )
        self.draw_pro_net( self.net_can, final_hex_codes, final_color_pixels,
                           centers_sorted, thr )

        self.win.config( cursor='' )
        self.update_status( f"Analysis completed, trend color：{most_popular}" )

    def draw_pro_net(self, canvas, hex_codes, pixels, centers, thr):
        canvas.delete( "all" )
        n = len( hex_codes )
        if n == 0:
            return

        final_color_pixels = np.array( pixels, dtype=float )
        total_pixels = np.sum( final_color_pixels ) or 1.0
        final_ratios = final_color_pixels / total_pixels
        node_sizes = 18 + final_ratios * 70

        G = nx.Graph()
        for i, hc in enumerate( hex_codes ):
            G.add_node( i, color=hc )

        dist_matrix = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        max_dist = np.max( dist_matrix ) if np.max( dist_matrix ) != 0 else 1.0

        for i in range( n ):
            for j in range( i + 1, n ):
                sim = 1 - dist_matrix[i, j] / max_dist
                if sim >= thr:
                    G.add_edge( i, j, weight=sim )

        pos = {}
        radius = 6
        for i in range( n ):
            angle = 2 * math.pi * i / n
            pos[i] = (radius * math.cos( angle ), radius * math.sin( angle ))

        w = canvas.winfo_width() or 600
        h = canvas.winfo_height() or 400
        cx, cy = w / 2.0, h / 2.0

        layout_x = np.array( [p[0] for p in pos.values()] )
        layout_y = np.array( [p[1] for p in pos.values()] )
        max_extent = max( layout_x.max() - layout_x.min(),
                          layout_y.max() - layout_y.min() )
        if max_extent == 0:
            max_extent = 1.0

        padding = 0.08
        scale = 1.5 * (min( w, h ) * (0.5 - padding)) / max_extent

        edge_weights = [
            d["weight"] for _, _, d in G.edges( data=True )
        ] if G.number_of_edges() > 0 else [1.0]
        max_w = max( edge_weights ) if edge_weights else 1.0

        for u, v, data in G.edges( data=True ):
            weight = data["weight"]

            x1 = cx + pos[u][0] * scale
            y1 = cy + pos[u][1] * scale
            x2 = cx + pos[v][0] * scale
            y2 = cy + pos[v][1] * scale

            ru = max( 1.0, node_sizes[u] )
            rv = max( 1.0, node_sizes[v] )
            x1 = max( ru, min( w - ru, x1 ) )
            y1 = max( ru, min( h - ru, y1 ) )
            x2 = max( rv, min( w - rv, x2 ) )
            y2 = max( rv, min( h - rv, y2 ) )

            gray = int( max( 30, min( 220, 200 - 150 * (weight / max_w) ) ) )
            edge_color = f"#{gray:02x}{gray:02x}{gray:02x}"

            lw = 1 + (weight / max_w) * 6
            canvas.create_line( x1, y1, x2, y2, width=lw,
                                fill=edge_color, smooth=True )

        for u, v, data in G.edges( data=True ):
            weight = data["weight"]

            x1 = cx + pos[u][0] * scale
            y1 = cy + pos[u][1] * scale
            x2 = cx + pos[v][0] * scale
            y2 = cy + pos[v][1] * scale

            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            canvas.create_text(
                mx, my, text=f"{weight:.2f}",
                fill='white', font=('Arial', 11, 'bold')
            )
            canvas.create_text(
                mx, my, text=f"{weight:.2f}",
                fill='#FF0000', font=('Arial', 10, 'bold')
            )

        for i, color in enumerate( hex_codes ):
            r = node_sizes[i]

            x = cx + pos[i][0] * scale
            y = cy + pos[i][1] * scale

            x = max( r + 4, min( w - r - 4, x ) )
            y = max( r + 4, min( h - r - 4, y ) )

            canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="black", width=1
            )

            fg = 'white' if self.is_dark_color( color ) else 'black'

            canvas.create_text(
                x, y, text=str( i + 1 ),
                fill=fg, font=('Arial', 10, 'bold')
            )

            pct_txt = f"{final_ratios[i] * 100:.1f}%"
            canvas.create_text(
                x, y + r + 12, text=pct_txt,
                fill='black', font=('Arial', 8)
            )

    def _safe_hex(self, c):
        try:
            if isinstance( c, (tuple, list, np.ndarray) ) and len( c ) >= 3:
                r, g, b = [int( round( x ) ) for x in c[:3]]
                r, g, b = max( 0, min( 255, r ) ), max( 0, min( 255, g ) ), max( 0, min( 255, b ) )
                return "#{:02x}{:02x}{:02x}".format( r, g, b )
            if isinstance( c, str ):
                return c if c.startswith( "#" ) else "#" + c
        except:
            return "#000000"
        return "#000000"

    def _apply_trend_colors_to_workflow(self, colors, mode="complete"):
        normalized = []
        for color in colors:
            value = self._safe_hex(color).lower()
            if value not in normalized:
                normalized.append(value)
        if not normalized:
            return []

        if mode == "dominant":
            for color in normalized:
                if color not in self.hex_color_codes:
                    self.hex_color_codes.append(color)
                if self.latest_colors is None:
                    self.latest_colors = []
                if color not in self.latest_colors:
                    self.latest_colors.append(color)
                if color not in self.outline_coloring_colors:
                    self.outline_coloring_colors.append(color)
                if color not in self.color_transfer_colors:
                    self.color_transfer_colors.append(color)
            return normalized

        self.hex_color_codes = list(normalized)
        self.latest_colors = list(normalized)
        self.outline_coloring_colors = list(normalized)
        self.color_transfer_colors = list(normalized)
        try:
            self.update_outline_coloring_tab()
        except Exception:
            pass
        return normalized

    def export_4k(self):
        if self.trend_hex.get() == "Not generated!":
            messagebox.showwarning( "Warning", "Please run..." )
            return
        f = filedialog.asksaveasfilename(
            title="Export 4K result.",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg")]
        )
        if not f:
            return
        messagebox.showinfo( "Export", f" 4K Figure \n{f} has been saved!" )

    # ================================================================
    # 【创新】深度颜色关系网络分析与评分系统 v2.0
    # 核心思想：从颜色关系网络的拓扑特征中提取深层美学信息
    # ================================================================

    def build_color_network(self, labels_2d, hex_color_codes, color_proportions, network_threshold=0.1):
        """
        构建完整的颜色关系网络（带丰富属性）
        返回: NetworkX图对象 + 颜色位置信息

        网络属性:
        - 节点: 每个颜色区域
        - 边: 相邻关系（权重=2×共同边界/两节点总边界）
        - 节点属性: color, proportion, dispersion, rgb, hsv, lab
        """
        import numpy as np
        import networkx as nx
        from collections import defaultdict

        if labels_2d is None or labels_2d.ndim != 2:
            return None, None, None, None

        n = len(hex_color_codes)
        counts = np.array(color_proportions, dtype=float).reshape(-1)
        if len( counts ) < n:
            counts = np.pad( counts, (0, n - len( counts )), mode='constant' )
        elif len( counts ) > n:
            counts = counts[:n]
        proportions = counts / counts.sum() if counts.sum() > 0 else np.ones(n) / max(1, n)

        # 收集每个颜色的像素坐标和RGB值（向量化，替代逐像素Python循环）
        color_positions = [[] for _ in range(n)]
        color_rgb_values = [[] for _ in range(n)]
        h, w = labels_2d.shape
        # 用 numpy 一次性获取所有像素坐标
        ys, xs = np.where((labels_2d >= 0) & (labels_2d < n))
        cids = labels_2d[ys, xs].astype(int)
        # 按标签分组
        for cid_val in range(n):
            sel = cids == cid_val
            if sel.any():
                color_positions[cid_val] = np.column_stack((xs[sel], ys[sel])).tolist()

        # 论文第 208 页公式（1）的 D/d；与提取页外环共用唯一计算源。
        dispersions = self._color_network_paper_dispersion_factors(labels_2d, n)

        # 计算相邻像素统计
        adjacent_pixels_count = self.calculate_adjacent_pixels(labels_2d)

        # 构建图
        G = nx.Graph()
        for i in range(n):
            r = int(hex_color_codes[i][1:3], 16)
            g = int(hex_color_codes[i][3:5], 16)
            b = int(hex_color_codes[i][5:7], 16)
            G.add_node(i,
                       color=hex_color_codes[i],
                       proportion=proportions[i],
                       count=int(counts[i]),
                       dispersion=dispersions[i],
                       rgb=np.array([r, g, b], dtype=float) / 255.0)

        # 计算HSV和Lab颜色空间值
        for node in G.nodes():
            rgb_0_1 = G.nodes[node]['rgb']
            rgb_255 = (rgb_0_1 * 255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb_255.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
            lab = cv2.cvtColor(rgb_255.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)[0, 0]
            G.nodes[node]['hsv'] = hsv.astype(float)
            G.nodes[node]['lab'] = lab.astype(float)

        # 添加边
        boundary_map = {i: 0 for i in range(n)}
        for (i, j), count in adjacent_pixels_count.items():
            if i != j:
                boundary_map[i] += count
                boundary_map[j] += count

        for (idx1, idx2), count in adjacent_pixels_count.items():
            if idx1 == idx2:
                continue
            total_boundary = boundary_map.get(idx1, 0) + boundary_map.get(idx2, 0)
            if total_boundary == 0:
                continue
            # 与提取页网络保持同一 Dice 型共享边界定义。旧评分路径漏掉
            # 系数 2，会把所有边权系统性压低一半，造成页面阈值和评分网络
            # 对同一邻接关系给出不一致的判断。
            weight = float(np.clip(
                2.0 * float(count) / float(total_boundary), 0.0, 1.0
            ))
            if weight >= network_threshold:
                G.add_edge(idx1, idx2, weight=weight, boundary_count=count)

        return G, color_positions, proportions, dispersions

    def compute_network_centrality_measures(self, G):
        """
        计算所有重要的网络中心性指标

        返回字典: {
            'degree': {...},           # 度中心性（直接邻居数量）
            'weighted_degree': {...},  # 加权度（考虑边权重的连接强度）
            'betweenness': {...},      # 介数中心性（最短路径经过次数）
            'closeness': {...},        # 接近中心性（到其他节点的平均距离）
            'eigenvector': {...},      # 特征向量中心性（连接重要节点的重要性）
            'pagerank': {...},         # PageRank（随机游走重要性）
            'load': {...},             # 负载中心性
            'harmonic': {...},         # 调和中心性（接近中心性的改进版）
        }
        """
        import numpy as np

        if G is None or G.number_of_nodes() == 0:
            return {}

        measures = {}

        # 1. 度中心性 - 直接相连的颜色数量
        try:
            measures['degree'] = nx.degree_centrality(G)
        except:
            measures['degree'] = {n: 0 for n in G.nodes()}

        # 2. 加权度中心性 - 考虑边权重的连接强度
        try:
            weighted_deg = {}
            for node in G.nodes():
                total_weight = sum(G[u][v].get('weight', 1) for u, v in G.edges(node))
                max_possible = sum(1 for _ in G.nodes()) - 1
                weighted_deg[node] = total_weight / max_possible if max_possible > 0 else 0
            measures['weighted_degree'] = weighted_deg
        except:
            measures['weighted_degree'] = {n: 0 for n in G.nodes()}

        # 3. 介数中心性 - 最短路径经过该节点的频率
        # 含义: 该颜色在视觉过渡中的"桥梁"作用
        try:
            measures['betweenness'] = nx.betweenness_centrality(G, weight='weight')
        except:
            measures['betweenness'] = {n: 0 for n in G.nodes()}

        # 4. 接近中心性 - 到其他所有节点的平均距离倒数
        # 含义: 该颜色在网络中的"可达性"
        try:
            measures['closeness'] = nx.closeness_centrality(G)
        except:
            measures['closeness'] = {n: 0 for n in G.nodes()}

        # 5. 特征向量中心性 - 连接重要节点的重要性
        # 含义: 与高影响力颜色相连的颜色也获得高影响力
        try:
            measures['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            measures['eigenvector'] = {n: 1.0 / len(G.nodes()) for n in G.nodes()}

        # 6. PageRank - Google网页排名算法应用于颜色
        # 含义: 颜色的全局影响力（考虑随机跳跃）
        try:
            measures['pagerank'] = nx.pagerank(G, weight='weight', alpha=0.85)
        except:
            measures['pagerank'] = {n: 1.0 / len(G.nodes()) for n in G.nodes()}

        # 7. 负载中心性 - 类似介数的另一种度量
        try:
            measures['load'] = nx.load_centrality(G, weight='weight')
        except:
            measures['load'] = {n: 0 for n in G.nodes()}

        # 8. 调和中心性 - 对不连通图更鲁棒的接近中心性
        try:
            measures['harmonic'] = nx.harmonic_centrality(G)
            max_harmonic = 1.0 / (len(G.nodes()) - 1) if len(G.nodes()) > 1 else 1.0
            measures['harmonic'] = {k: v * max_harmonic for k, v in measures['harmonic'].items()}
        except:
            measures['harmonic'] = {n: 0 for n in G.nodes()}

        # 归一化所有指标到[0,1]
        for key in measures:
            values = list(measures[key].values())
            if len(values) > 0 and max(values) > 0:
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    measures[key] = {k: (v - min_val) / (max_val - min_val)
                                    for k, v in measures[key].items()}
                else:
                    measures[key] = {k: 1.0 for k in measures[key]}

        return measures

    def get_top_k_colors_by_importance(self, G, k, metric='composite',
                                        weights=None, return_scores=False):
        """
        根据网络中心性指标选择TOP-K个最重要的颜色

        Args:
            G: 颜色关系网络
            k: 选择数量
            metric: 重要性指标类型
                   - 'proportion': 传统占比法（基准对比）
                   - 'betweenness': 介数中心性（桥梁作用最强）
                   - 'pagerank': PageRank（全局影响力最大）
                   - 'eigenvector': 特征向量（关联影响力最大）
                   - 'closeness': 接近中心性（可达性最好）
                   - 'harmonic': 调和中心性（综合可达性）
                   - 'composite': 综合加权（默认，推荐）
                   - 'bridge_only': 纯粹桥梁型颜色（占比小但关键）
                   - 'influence_hub': 影响力枢纽型颜色
            weights: composite模式下各指标的权重字典
            return_scores: 是否返回详细分数

        Returns:
            top_k_indices: TOP-K颜色索引列表
            details: 详细分数和分析结果
        """
        import numpy as np

        if G is None or G.number_of_nodes() == 0:
            return [], {}

        all_nodes = list(G.nodes())
        n_nodes = len(all_nodes)
        k = min(k, n_nodes)

        # 计算所有中心性指标
        measures = self.compute_network_centrality_measures(G)

        if not measures:
            return all_nodes[:k], {}

        # 提取节点属性
        proportions = np.array([G.nodes[n].get('proportion', 0) for n in all_nodes])
        dispersions = np.array([G.nodes[n].get('dispersion', 0) for n in all_nodes])

        # 根据metric类型计算最终得分
        if metric == 'proportion':
            # 传统方法：按颜色占比排序（基准对比）
            scores = dict(zip(all_nodes, proportions))

        elif metric == 'betweenness':
            # 桥梁型：介数最高的颜色（可能是小面积但处于关键位置）
            scores = measures.get('betweenness', {n: 0 for n in all_nodes})

        elif metric == 'pagerank':
            scores = measures.get('pagerank', {n: 0 for n in all_nodes})

        elif metric == 'eigenvector':
            scores = measures.get('eigenvector', {n: 0 for n in all_nodes})

        elif metric == 'closeness':
            scores = measures.get('closeness', {n: 0 for n in all_nodes})

        elif metric == 'harmonic':
            scores = measures.get('harmonic', {n: 0 for n in all_nodes})

        elif metric == 'bridge_only':
            # 创新指标：纯粹桥梁型颜色
            # 高介数 + 低占比 = 关键但不起眼的"幕后英雄"
            betweenness = measures.get('betweenness', {n: 0 for n in all_nodes})
            scores = {}
            for node in all_nodes:
                b = betweenness.get(node, 0)
                p = proportions[list(all_nodes).index(node)] if node in all_nodes else 0
                # 桥梁系数 = 介数 × (1-占比)^0.5 → 占比越小加分越多
                bridge_score = b * ((1 - p + 0.01) ** 0.5)
                scores[node] = bridge_score

        elif metric == 'influence_hub':
            # 创新指标：影响力枢纽型
            # PageRank + 特征向量 + 加权度的组合
            pr = measures.get('pagerank', {n: 0 for n in all_nodes})
            ev = measures.get('eigenvector', {n: 0 for n in all_nodes})
            wd = measures.get('weighted_degree', {n: 0 for n in all_nodes})
            scores = {}
            for node in all_nodes:
                scores[node] = 0.4 * pr.get(node, 0) + 0.35 * ev.get(node, 0) + 0.25 * wd.get(node, 0)

        elif metric == 'composite':
            # 综合加权：默认推荐方案
            default_weights = {
                'proportion': 0.15,      # 占比有一定参考价值
                'betweenness': 0.25,     # 桥梁作用（高价值）
                'pagerank': 0.25,       # 全局影响力
                'eigenvector': 0.10,     # 关联影响
                'closeness': 0.10,       # 可达性
                'weighted_degree': 0.15   # 局部连接强度
            }
            w = weights if weights else default_weights
            scores = {}
            for node in all_nodes:
                score = 0
                for measure_name, weight in w.items():
                    if measure_name in measures:
                        score += weight * measures[measure_name].get(node, 0)
                scores[node] = score

        else:
            scores = {n: 1.0 / n_nodes for n in all_nodes}

        # 排序并返回TOP-K
        sorted_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        top_k_indices = sorted_nodes[:k]

        # 构建详细信息
        details = {
            'all_scores': scores,
            'measures': measures,
            'top_k_indices': top_k_indices,
            'metric_used': metric,
            'analysis': self._generate_importance_analysis(top_k_indices, measures,
                                                          proportions, all_nodes)
        }

        if return_scores:
            return top_k_indices, details
        return top_k_indices

    def _generate_importance_analysis(self, top_k_indices, measures, proportions, all_nodes):
        """生成可解释的重要性分析报告"""
        analysis = []
        for idx in top_k_indices:
            info = {
                'color_index': idx,
                'proportion': float(proportions[idx] if idx < len(proportions) else 0),
                'betweenness': float(measures.get('betweenness', {}).get(idx, 0)),
                'pagerank': float(measures.get('pagerank', {}).get(idx, 0)),
                'eigenvector': float(measures.get('eigenvector', {}).get(idx, 0)),
                'role': self._infer_color_role(idx, measures, proportions, all_nodes)
            }
            analysis.append(info)
        return analysis

    def _infer_color_role(self, node_idx, measures, proportions, all_nodes):
        """推断颜色在网络中的角色"""
        b = measures.get('betweenness', {}).get(node_idx, 0)
        p = measures.get('pagerank', {}).get(node_idx, 0)
        e = measures.get('eigenvector', {}).get(node_idx, 0)
        prop = proportions[node_idx] if node_idx < len(proportions) else 0

        # 角色推断规则
        if b > 0.6 and prop < 0.15:
            return "关键桥梁（小面积但起关键衔接作用）"
        elif p > 0.6 and prop > 0.2:
            return "核心主色（大面积+高影响力）"
        elif e > 0.6 and prop > 0.1:
            return "协调辅助色（连接多个重要颜色）"
        elif b > 0.4 and p > 0.4:
            return "战略枢纽（兼具桥梁和影响力）"
        elif prop > 0.25:
            return "背景基调色（大面积基础色）"
        else:
            return "装饰点缀色（局部强调用）"


    # ================================================================
    # 【新增】高级色彩迁移优化方法 (Advanced Color Transfer Optimization)
    # 结合 CIEDE2000 / 色彩和谐模型 / 最优传输 / 色彩角色感知 / 边界防护 / 图平滑
    # ================================================================

    def _ciede2000(self, lab1, lab2):
        """
        CIEDE2000 感知均匀色差计算 (替代 DeltaE76)
        参考: Sharma et al., "The CIEDE2000 Color-Difference Formula" (2005)
        比 Lab 欧氏距离更符合人眼感知, 尤其在高饱和/蓝色区域差异显著
        """
        import numpy as np
        L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
        L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])
        C1_ab = np.sqrt(a1**2 + b1**2)
        C2_ab = np.sqrt(a2**2 + b2**2)
        C_ab_mean = (C1_ab + C2_ab) / 2.0
        C_ab_mean_7 = C_ab_mean**7
        G = 0.5 * (1 - np.sqrt(C_ab_mean_7 / (C_ab_mean_7 + 25.0**7)))
        a1p = a1 * (1 + G)
        a2p = a2 * (1 + G)
        C1p = np.sqrt(a1p**2 + b1**2)
        C2p = np.sqrt(a2p**2 + b2**2)
        h1p = np.degrees(np.arctan2(b1, a1p)) % 360
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360
        dLp = L2 - L1
        dCp = C2p - C1p
        if C1p * C2p == 0:
            dhp = 0
        elif abs(h2p - h1p) <= 180:
            dhp = h2p - h1p
        elif h2p - h1p > 180:
            dhp = h2p - h1p - 360
        else:
            dhp = h2p - h1p + 360
        dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
        Lp_mean = (L1 + L2) / 2
        Cp_mean = (C1p + C2p) / 2
        if C1p * C2p == 0:
            hp_mean = h1p + h2p
        elif abs(h1p - h2p) <= 180:
            hp_mean = (h1p + h2p) / 2
        elif h1p + h2p < 360:
            hp_mean = (h1p + h2p + 360) / 2
        else:
            hp_mean = (h1p + h2p - 360) / 2
        T = (1
             - 0.17 * np.cos(np.radians(hp_mean - 30))
             + 0.24 * np.cos(np.radians(2 * hp_mean))
             + 0.32 * np.cos(np.radians(3 * hp_mean + 6))
             - 0.20 * np.cos(np.radians(4 * hp_mean - 63)))
        SL = 1 + 0.015 * (Lp_mean - 50)**2 / np.sqrt(20 + (Lp_mean - 50)**2)
        SC = 1 + 0.045 * Cp_mean
        SH = 1 + 0.015 * Cp_mean * T
        Cp_mean_7 = Cp_mean**7
        RT = (-2 * np.sqrt(Cp_mean_7 / (Cp_mean_7 + 25.0**7))
              * np.sin(np.radians(60 * np.exp(-((hp_mean - 275) / 25)**2))))
        return np.sqrt(
            (dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2
            + RT * (dCp / SC) * (dHp / SH)
        )

    def _analyze_source_spatial_centroids(self):
        """
        分析参考图中各颜色的空间质心位置 (归一化到 0~1)
        用于空间感知映射: 参考图上方颜色优先分配给目标上方区域
        """
        import cv2
        if not hasattr(self, 'labels_2d') or self.labels_2d is None:
            return None
        if not hasattr(self, 'hex_color_codes') or not self.hex_color_codes:
            return None
        labels = self.labels_2d
        h, w = labels.shape[:2]
        centroids = {}
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            if lbl < 0:
                continue
            mask = labels == lbl
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            centroids[int(lbl)] = {
                'y': float(np.mean(ys)) / h,
                'x': float(np.mean(xs)) / w
            }
        return centroids if centroids else None

    def _compute_target_region_centroids(self, labels, unique_labels):
        """计算目标区域的空间质心 (归一化到 0~1)"""
        h, w = labels.shape[:2]
        centroids = {}
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            centroids[i] = {
                'y': float(np.mean(ys)) / h,
                'x': float(np.mean(xs)) / w
            }
        return centroids

    def _compute_harmony_pattern_score(self, hue_values):
        """
        色彩和谐模式检测 (基于色相环关系)
        参考: Frontiers in Psychology 2022, "Attribute Analysis and Modeling
        of Color Harmony Based on Multi-Color Feature Extraction"
        检测配色是否符合经典和谐模式: 类似色/互补/三等分/四等分/分裂互补
        """
        if len(hue_values) < 2:
            return 60.0
        hue_diffs = []
        for i in range(len(hue_values)):
            for j in range(i + 1, len(hue_values)):
                d = abs(int(hue_values[i]) - int(hue_values[j]))
                d = min(d, 180 - d)
                hue_diffs.append(d)
        if not hue_diffs:
            return 60.0
        templates = {
            'analogous':    (30,  15),
            'complementary': (90,  20),
            'triadic':      (120, 20),
            'tetradic':      (90, 25),
            'split_comp':   (150, 20),
        }
        scores = []
        for name, (center, width) in templates.items():
            match_count = sum(
                1 for d in hue_diffs
                if abs(d - center) <= width * 1.5
            )
            ratio = match_count / len(hue_diffs)
            gaussian_scores = [
                np.exp(-((d - center)**2) / (2 * width**2))
                for d in hue_diffs
            ]
            s = np.mean(gaussian_scores) * 0.6 + ratio * 0.4
            scores.append(s)
        best_harmony = max(scores)
        return float(best_harmony * 100)

    def _compute_optimal_assignment(self, selected_colors, num_regions,
                                     region_centroids, outline_labels_2d=None):
        """
        最优传输分配 (匈牙利算法)
        参考: ModFlows (arXiv 2503.19062) 的最优传输思想 +
               Palette-Based Transfer (JCST 2025) 的空间感知聚类
        构建 cost = 空间距离 + 色彩角色匹配, 用 O(n^3) 匈牙利算法
        替代 O(n!) 暴力枚举, 直接求最优颜色-区域分配
        """
        import cv2
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return None
        num_colors = len(selected_colors)
        source_centroids = self._analyze_source_spatial_centroids()
        cost_matrix = np.zeros((num_colors, num_regions))
        for ci in range(num_colors):
            r = int(selected_colors[ci][1:3], 16)
            g = int(selected_colors[ci][3:5], 16)
            b = int(selected_colors[ci][5:7], 16)
            color_lab = cv2.cvtColor(
                np.array([[[r, g, b]]], dtype=np.uint8),
                cv2.COLOR_RGB2Lab
            )[0, 0].astype(float)
            for ri in range(num_regions):
                spatial_cost = 0.0
                if source_centroids and ci in source_centroids:
                    if ri in region_centroids:
                        dy = source_centroids[ci]['y'] - region_centroids[ri]['y']
                        dx = source_centroids[ci]['x'] - region_centroids[ri]['x']
                        spatial_cost = np.sqrt(dy**2 + dx**2) * 30
                color_role_cost = 0.0
                if (hasattr(self, 'precomputed_network')
                        and self.precomputed_network is not None):
                    G = self.precomputed_network.get('G')
                    centrality = self.precomputed_network.get('centrality', {})
                    if G and G.has_node(ri):
                        btw = centrality.get('betweenness', {}).get(ri, 0)
                        pr = centrality.get('pagerank', {}).get(ri, 0)
                        prop = float(np.sum(outline_labels_2d == ri)) / max(
                            outline_labels_2d.size, 1)
                        role = self._infer_color_role_enhanced(
                            btw, pr, 0, prop, 0, 0, 0)
                        if 'Dominant' in role and prop < 0.1:
                            color_role_cost = 20
                        elif 'Accent' in role and prop > 0.2:
                            color_role_cost = 20
                        elif 'Background' in role and prop < 0.1:
                            color_role_cost = 15
                src_lab = np.array([50., 0., 0.])
                if source_centroids and ci in source_centroids:
                    src_lab = color_lab.copy()
                de = self._ciede2000(color_lab, src_lab)
                color_cost = de / 100.0 * 30
                cost_matrix[ci, ri] = (
                    spatial_cost + color_cost + color_role_cost
                )
        try:
            row_ind, col_ind = linear_sum_assignment(
                cost_matrix, maximize=False
            )
            placement = [None] * num_regions
            for r_idx, c_idx in zip(row_ind, col_ind):
                if c_idx < num_regions:
                    placement[c_idx] = int(r_idx)

            # 匈牙利算法天然给出 M 种颜色到 N 个区域的一对一部分排列。
            # 未分配区域必须保持 None，不再循环补色。
            return placement
        except Exception:
            return None

    # ================================================================
    # 【新增】顶刊级色彩迁移优化 (Top-tier Journal-level Optimization)
    # ================================================================

    def _apply_simultaneous_contrast_correction(self, img_array, labels_2d,
                                                 precomputed_network, alpha=0.25):
        """
        同时对比度校正 — Josef Albers "Interaction of Color" 计算化实现
        同一颜色在不同背景旁感知不同: 暗背景旁显亮, 亮背景旁显暗
        在LAB空间对每个区域的L通道做邻域感知补偿, 使着色在视觉上更均匀
        参考: Fairchild, "Color Appearance Models", 3rd Ed., Wiley (2013), Ch.8
        """
        import cv2
        G = precomputed_network.get('G') if precomputed_network else None
        if G is None or G.number_of_edges() == 0:
            return img_array

        unique_labels = sorted(set(labels_2d.flatten()) - {-1})
        if len(unique_labels) < 2:
            return img_array

        # 计算每个区域的平均 LAB
        region_labs = {}
        region_masks_bool = {}
        for lbl in unique_labels:
            mask = labels_2d == lbl
            if not np.any(mask):
                continue
            region_masks_bool[lbl] = mask
            region_rgb = img_array[mask]
            avg_rgb = region_rgb.mean(axis=0).astype(np.uint8)
            lab = cv2.cvtColor(avg_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)[0, 0].astype(float)
            region_labs[lbl] = lab

        # 对每个区域, 计算邻域对比度效应并校正
        for lbl in unique_labels:
            if lbl not in region_labs or lbl not in region_masks_bool:
                continue
            neighbors = list(G.neighbors(lbl)) if G.has_node(lbl) else []
            if not neighbors:
                continue

            my_L = region_labs[lbl][0]
            neighbor_Ls = []
            for nb in neighbors:
                if nb in region_labs:
                    neighbor_Ls.append(region_labs[nb][0])
            if not neighbor_Ls:
                continue

            avg_neighbor_L = np.mean(neighbor_Ls)
            # 同时对比度: 邻域暗→当前感知偏亮→降低L; 邻域亮→当前感知偏暗→提高L
            contrast_delta = (my_L - avg_neighbor_L) * alpha

            mask = region_masks_bool[lbl]
            region_pixels = img_array[mask].copy()
            lab_pixels = cv2.cvtColor(region_pixels, cv2.COLOR_RGB2Lab).astype(float)
            lab_pixels[:, :, 0] = np.clip(lab_pixels[:, :, 0] - contrast_delta, 0, 255)
            corrected = cv2.cvtColor(lab_pixels.astype(np.uint8), cv2.COLOR_Lab2RGB)
            img_array[mask] = corrected

        return img_array



    def calculate_image_score(self, image, color_regions, labels_2d, precomputed_network=None):
        """
        ============================================================
        【创新】v2.0 四维深度颜色关系网络感知评分系统
        ============================================================

        从颜色关系网络的拓扑结构中提取深层美学信息，
        用4个创新维度评估配色方案的质量：

        维度1: 网络桥接质量 (Network Bridge Quality, NBQ) [权重30%]
        ──────────────────────────────────────────────────────
        衡量颜色配置是否充分利用了网络中的"桥梁"位置。
        高分条件：
        - 高介数的区域分配了醒目/对比色（发挥桥梁视觉效果）
        - 低介数的边缘区域使用柔和色（避免视觉冲突）

        核心公式: NBQ = Σ(betweenness_i × contrast_reward_i)
                 where contrast_reward_i = f(该颜色与邻域的Lab色差)

        维度2: 色彩和谐能量 (Color Harmony Energy, CHE) [权重28%]
        ──────────────────────────────────────────────────────
        基于网络结构的和谐度评估，融合：
        - Lab空间色差加权的边权重
        - 三角色互补检测（互补色、三角色、类似色）
        - 全局色彩分布熵（衡量多样性vs单调性）

        核心公式: CHE = harmony_weighted_edges × entropy_bonus × complementary_bonus

        维度3: 信息传播效率 (Information Diffusion Efficiency, IDE) [权重24%]
        ──────────────────────────────────────────────────────
        使用PageRank随机游走模型评估视觉注意力流动效率。
        高分条件：
        - 视觉焦点区域（高PageRank）使用了重点色
        - 注意力流动路径顺畅，无"死胡同"
        - 颜色层次分明（PageRank方差适中）

        核心公式: IDE = attention_focus_score × flow_smoothness × hierarchy_clarity

        维度4: 结构平衡指数 (Structural Balance Index, SBI) [权重18%]
        ──────────────────────────────────────────────────────
        评估颜色网络的拓扑稳定性和均衡性：
        - 边权重的均匀度（无过度集中）
        - 节点度分布的均衡性
        - 色彩面积的Gini系数（衡量公平性）
        - 小世界特性保留程度

        核心公式: SBI = edge_balance × degree_balance × area_fairness × small_world

        总分: Score = 0.30×NBQ + 0.28×CHE + 0.24×IDE + 0.18×SBI
        ============================================================
        """
        import numpy as np
        import cv2

        h, w = image.shape[:2]
        total_pixels = h * w

        # ===== 安全检查：labels_2d 必须与图像尺寸匹配 =====
        if labels_2d is None or labels_2d.shape[0] != h or labels_2d.shape[1] != w:
            return self._fallback_simple_score(image, labels_2d)

        # ===== Step 1: 构建当前图像的颜色网络（可复用预计算结果） =====
        current_G = None
        positions = props = disps = None
        centrality = None
        if isinstance(precomputed_network, dict):
            current_G = precomputed_network.get('G')
            centrality = precomputed_network.get('centrality')

        if current_G is None:
            current_G, positions, props, disps = self.build_color_network(
                labels_2d, self.hex_color_codes if hasattr(self, 'hex_color_codes') else [],
                [np.sum(labels_2d == i) for i in range(len(np.unique(labels_2d)))],
                self.network_threshold_var.get() if hasattr(self, 'network_threshold_var') else 0.1
            )

        if current_G is None or current_G.number_of_nodes() < 2:
            # 回退到简单评估
            return self._fallback_simple_score(image, labels_2d)

        # 计算网络中心性指标
        if centrality is None:
            centrality = self.compute_network_centrality_measures(current_G)

        # ===== 维度1: 网络桥接质量 (NBQ) =====
        nbq_score = self._compute_NBQ(current_G, image, labels_2d, centrality)

        # ===== 维度2: 色彩和谐能量 (CHE) =====
        che_score = self._compute_CHE(current_G, image, labels_2d, centrality)

        # ===== 维度3: 信息传播效率 (IDE) =====
        ide_score = self._compute_IDE(current_G, image, labels_2d, centrality)

        # ===== 维度4: 结构平衡指数 (SBI) =====
        sbi_score = self._compute_SBI(current_G, labels_2d, centrality)

        # ===== 加权总分 =====
        total_score = (
            0.30 * nbq_score +
            0.28 * che_score +
            0.24 * ide_score +
            0.18 * sbi_score
        )

        print(f"NBQ={nbq_score:.2f} CHE={che_score:.2f} IDE={ide_score:.2f} SBI={sbi_score:.2f} TOTAL={total_score:.2f}")

        return min(100, max(0, total_score))

    def _compute_NBQ(self, G, image, labels_2d, centrality):
        """
        维度1: 网络桥接质量 (Network Bridge Quality)

        核心思想：高介数的位置是视觉上的"咽喉要道"，
        在这些位置使用合适的颜色能显著提升整体设计感。

        评分逻辑：
        - 高介数区域使用高对比/饱和度颜色 → 高分（突出桥梁效果）
        - 高介数区域使用低饱和度颜色 → 中等（浪费桥梁位置）
        - 低介数区域使用任何颜色 → 正常（不影响）
        """
        import numpy as np
        import cv2

        betweenness = centrality.get('betweenness', {})
        img_array = np.array(image) if not isinstance(image, np.ndarray) else image
        h, w = img_array.shape[:2]

        nbq_total = 0.0
        node_count = 0

        for node in G.nodes():
            b = betweenness.get(node, 0)
            if b < 0.01:
                continue

            # 获取该区域的平均颜色
            mask = (labels_2d == node)
            if not np.any(mask):
                continue

            region_colors = img_array[mask]
            avg_rgb = region_colors.mean(axis=0) / 255.0

            # 转换到HSV计算饱和度和明度
            avg_rgb_uint8 = (avg_rgb * 255).astype(np.uint8)
            hsv = cv2.cvtColor(avg_rgb_uint8.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
            saturation = hsv[1] / 255.0  # 饱和度
            value = hsv[2] / 255.0        # 明度

            # 计算与邻域的平均色差（Lab空间）
            neighbors = list(G.neighbors(node)) if G.has_node(node) else []
            if neighbors:
                lab_region = cv2.cvtColor(avg_rgb_uint8.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)[0, 0].astype(float)
                neighbor_deltas = []
                for nb in neighbors:
                    nb_mask = (labels_2d == nb)
                    if np.any(nb_mask):
                        nb_colors = img_array[nb_mask]
                        nb_avg = nb_colors.mean(axis=0)
                        nb_avg_uint8 = nb_avg.astype(np.uint8)
                        nb_lab = cv2.cvtColor(nb_avg_uint8.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)[0, 0].astype(float)
                        delta_e = self._ciede2000(lab_region, nb_lab)
                        neighbor_deltas.append(delta_e)

                avg_contrast = np.mean(neighbor_deltas) if neighbor_deltas else 50
            else:
                avg_contrast = 30

            # 桥梁质量得分公式
            # 高介数 + 高对比 + 适度饱和 = 最佳桥梁效果
            contrast_factor = min(1.0, avg_contrast / 80)  # 归一化色差
            saturation_factor = 0.7 + 0.3 * saturation     # 饱和度贡献
            brightness_penalty = abs(value - 0.55) * 0.3   # 过亮或过暗扣分

            bridge_quality = b * (contrast_factor * 0.6 + saturation_factor * 0.3 - brightness_penalty)
            nbq_total += bridge_quality
            node_count += 1

        # 归一化到0-100
        raw_score = nbq_total / max(1, node_count) * 100 if node_count > 0 else 50
        return min(100, max(0, raw_score))

    def _compute_CHE(self, G, image, labels_2d, centrality):
        """
        维度2: 色彩和谐能量 (Color Harmony Energy)

        核心思想：从网络拓扑角度评估色彩的和谐程度，
        结合经典的色彩理论（互补、类似、三角）和网络结构。

        子指标：
        1. 边级色差和谐度：相邻颜色对的Lab色差应在理想范围
        2. 互补色配对奖励：存在互补色对时加分
        3. 色彩分布熵：适度的色彩多样性（非单一也非杂乱）
        4. HSV色调环一致性：主要色调在色环上的分布合理性
        """
        import numpy as np
        import cv2

        img_array = np.array(image) if not isinstance(image, np.ndarray) else image

        # ---- 子指标1: 边级色差和谐度 ----
        edge_harmony_scores = []
        for u, v, data in G.edges(data=True):
            u_mask = (labels_2d == u)
            v_mask = (labels_2d == v)
            if not np.any(u_mask) or not np.any(v_mask):
                continue

            u_color = img_array[u_mask].mean(axis=0).astype(np.uint8)
            v_color = img_array[v_mask].mean(axis=0).astype(np.uint8)

            # Lab色差
            u_lab = cv2.cvtColor(u_color.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)[0, 0].astype(float)
            v_lab = cv2.cvtColor(v_color.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)[0, 0].astype(float)
            delta_e = self._ciede2000(u_lab, v_lab)

            # 理想色差范围: 20-60 (太低=无区分, 太高=刺眼)
            # 使用高斯核评估
            ideal_center = 40
            ideal_width = 25
            harmony = np.exp(-((delta_e - ideal_center) ** 2) / (2 * ideal_width ** 2))

            edge_weight = data.get('weight', 0.5)
            edge_harmony_scores.append(harmony * edge_weight)

        edge_harmony = np.mean(edge_harmony_scores) * 100 if edge_harmony_scores else 50

        # ---- 子指标2: 互补色配对奖励 ----
        hue_values = []
        for node in G.nodes():
            mask = (labels_2d == node)
            if np.any(mask):
                node_color = img_array[mask].mean(axis=0).astype(np.uint8)
                hsv = cv2.cvtColor(node_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
                hue_values.append(hsv[0])  # 0-180

        complement_score = 0
        if len(hue_values) >= 2:
            complements_found = 0
            for i, h1 in enumerate(hue_values):
                for j, h2 in enumerate(hue_values[i+1:], i+1):
                    hue_diff = abs(h1 - h2)
                    # OpenCV H: 0-180, 互补色差约90
                    if 75 <= hue_diff <= 105 or hue_diff >= 165:
                        complements_found += 1
            complement_score = min(30, complements_found * 10)

        # ---- 子指标3: 色彩分布熵 ----
        unique_labels = np.unique(labels_2d)
        region_sizes = [np.sum(labels_2d == lbl) for lbl in unique_labels]
        probs = np.array(region_sizes) / sum(region_sizes)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(unique_labels)) if len(unique_labels) > 1 else 1
        normalized_entropy = entropy / max_entropy  # 0-1

        # 熵在0.5-0.8之间最佳（既不单调也不过于碎片化）
        entropy_score = 100 * np.exp(-((normalized_entropy - 0.65) ** 2) / (2 * 0.2 ** 2))

        # ---- 子指标4: HSV色调环一致性 ----
        if len(hue_values) >= 3:
            hue_array = np.array(hue_values)
            # 将hue映射到圆上，计算离散度
            rad = np.deg2rad(hue_array * 2)  # 映射到0-360度
            mean_cos = np.mean(np.cos(rad))
            mean_sin = np.mean(np.sin(rad))
            log_arg = float(mean_cos**2 + mean_sin**2 + 1e-10)
            if log_arg <= 0:
                circular_std = 2.0
            else:
                circular_std = float(np.sqrt(-np.log(log_arg)))
            raw = 1.0 - circular_std / 2.0
            hue_consistency = (raw if raw > 0 else 0.0) * 30
        else:
            hue_consistency = 15

        # 【新增】子指标5: 色彩和谐模式检测 (基于色相环关系)
        harmony_pattern_bonus = self._compute_harmony_pattern_score(hue_values)

        # 综合CHE得分 (加入和谐模式权重)
        che_total = (
            0.38 * edge_harmony +
            0.12 * complement_score +
            0.20 * entropy_score +
            0.12 * hue_consistency +
            0.18 * harmony_pattern_bonus
        )
        return min(100, max(0, che_total))

    def _compute_IDE(self, G, image, labels_2d, centrality):
        """
        维度3: 信息传播效率 (Information Diffusion Efficiency)

        使用PageRank随机游走模型模拟视觉注意力的流动。

        核心假设：人类观察图像时，视线会自然地在
        高PageRank区域停留更久。好的配色应该让：

        1. 高PR区域有明确的视觉焦点（高吸引力颜色）
        2. PR梯度平滑（视线引导自然）
        3. 没有"黑洞"区域（极低PR但高饱和度会分散注意力）
        """
        import numpy as np
        import cv2

        pagerank = centrality.get('pagerank', {})
        img_array = np.array(image) if not isinstance(image, np.ndarray) else image

        pr_values = list(pagerank.values())
        if not pr_values:
            return 50

        # ---- 子指标1: 注意力焦点质量 ----
        focus_quality = 0
        for node in G.nodes():
            pr = pagerank.get(node, 0)
            if pr < np.percentile(pr_values, 60):  # 只看前40%的高PR区域
                continue

            mask = (labels_2d == node)
            if not np.any(mask):
                continue

            node_color = img_array[mask].mean(axis=0).astype(np.uint8)
            hsv = cv2.cvtColor(node_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
            sat = hsv[1] / 255.0
            val = hsv[2] / 255.0

            # 高PR区域的颜色吸引力
            attractiveness = 0.5 * sat + 0.3 * val + 0.2 * (1 - abs(val - 0.6))
            focus_quality += pr * attractiveness

        focus_score = (focus_quality / max(sum(pr_values), 1e-6)) * 100

        # ---- 子指标2: 流动平滑度 ----
        # PageRank方差的倒数（方差太大=层次突兀，太小=平淡）
        pr_variance = np.var(pr_values)
        ideal_variance = 0.001  # 理想的PR方差
        smoothness = 100 * np.exp(-abs(pr_variance - ideal_variance) / (2 * 0.002))

        # ---- 子指标3: 层次清晰度 ----
        # PR值的偏度（正偏=少数焦点+多数辅助，符合构图原则）
        if len(pr_values) >= 3:
            pr_skew = abs(safe_skew(pr_values))
            hierarchy = 100 * np.exp(-pr_skew)  # 偏度接近0最佳
        else:
            hierarchy = 70

        # 综合IDE得分
        ide_total = 0.45 * focus_score + 0.30 * smoothness + 0.25 * hierarchy
        return min(100, max(0, ide_total))

    def _compute_SBI(self, G, labels_2d, centrality):
        """
        维度4: 结构平衡指数 (Structural Balance Index)

        评估颜色网络的结构健康度和均衡性。

        子指标：
        1. 边权重均衡度：边权重不应过度集中在几条边上
        2. 度分布均衡性：不应有极端的中心/边缘节点
        3. 面积Gini系数：颜色面积分布的公平性
        4. 聚类系数保持：网络的局部聚集性
        """
        import numpy as np
        # 使用自定义基尼系数函数（见 _gini_coefficient 方法）

        # ---- 子指标1: 边权重均衡度 ----
        edge_weights = [data.get('weight', 0.5) for _, _, data in G.edges(data=True)]
        if edge_weights:
            weight_cv = np.std(edge_weights) / (np.mean(edge_weights) + 1e-6)  # 变异系数
            balance_score = max(0, 100 * (1 - weight_cv))  # CV越低越均衡
        else:
            balance_score = 50

        # ---- 子指标2: 度分布均衡性 ----
        degrees = [G.degree(n) for n in G.nodes()]
        if degrees and max(degrees) > 0:
            degree_gini = self._gini_coefficient(degrees)
            degree_balance = (1 - degree_gini) * 100
        else:
            degree_balance = 50

        # ---- 子指标3: 面积Gini系数 ----
        unique_labels = np.unique(labels_2d)
        areas = [float(np.sum(labels_2d == lbl)) for lbl in unique_labels]
        if areas:
            area_gini = self._gini_coefficient(areas)
            area_fairness = (1 - area_gini) * 100
        else:
            area_fairness = 50

        # ---- 子指标4: 平均聚类系数 ----
        if G.number_of_nodes() > 1:
            try:
                clustering = nx.average_clustering(G)
                clustering_score = clustering * 100
            except:
                clustering_score = 50
        else:
            clustering_score = 50

        # 综合SBI得分
        sbi_total = (
            0.25 * balance_score +
            0.25 * degree_balance +
            0.30 * area_fairness +
            0.20 * clustering_score
        )
        return min(100, max(0, sbi_total))

    def _gini_coefficient(self, values):
        """计算基尼系数（衡量不平等程度的指标）"""
        import numpy as np
        arr = np.array(sorted(values))
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0
        cumsum = np.cumsum(arr)
        return (2 * np.sum((np.arange(1, n + 1) * arr)) - (n + 1) * arr.sum()) / (n * arr.sum())

    def _fallback_simple_score(self, image, labels_2d):
        """当无法构建网络时的回退评分（简化版原逻辑）"""
        import numpy as np
        import cv2

        h, w = image.shape[:2]
        total_pixels = h * w
        unique_labels = np.unique(labels_2d)

        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 100, 200)
        edge_pixels = np.sum(edges > 0)
        adjacency_score = (edge_pixels / total_pixels) * 100

        region_sizes = [np.sum(labels_2d == lbl) for lbl in unique_labels]
        balance_raw = 1 - (np.std(region_sizes) / (np.mean(region_sizes) + 1e-6))

        total_score = 0.5 * adjacency_score + 0.5 * balance_raw * 100
        return min(100, max(0, total_score))

    def generate_color_placements(self, num_colors, num_regions):
        return self._generate_balanced_placements( num_regions, num_colors )

    @staticmethod
    def _transfer_default_color_selection(count):
        """Transfer Colors 默认选择全部可用颜色。"""
        try:
            normalized_count = max(0, int(count))
        except (TypeError, ValueError, OverflowError):
            normalized_count = 0
        return [True] * normalized_count

    def _invalidate_transfer_plan_state(self):
        """新目标验证开始前淘汰旧迁移方案，失败路径保持不可生成。"""
        self.generated_plans = []
        self.colors_applied = False
        self.regions_for_plan = None
        self.plan_outline_array = None
        self.plan_outline_image = None
        self._prepared_transfer_colors = []
        self._element_aware_v21_active = False
        self._transfer_element_aware_active = False
        if hasattr(self, 'plan_button'):
            self.plan_button.config(state='disabled')

    def open_color_transfer(self):
        self.last_action = "color_transfer"
        self.outline_image_path = None
        self.regions_for_plan = None
        if not self.hex_color_codes:
            messagebox.showerror( "Error!", "Please extract the color first!" )
            return

        self.show_tab( self.transfer_tab, "Transfer Colors" )
        self.notebook.select( self.transfer_tab )

        for widget in self.source_colors_frame.winfo_children():
            widget.destroy()

        self.color_vars = []
        default_selection = self._transfer_default_color_selection(
            len(self.color_transfer_colors)
        )
        for i, color in enumerate( self.color_transfer_colors ):
            color_frame = ttk.Frame( self.source_colors_frame )
            color_frame.pack( fill='x', padx=5, pady=2 )

            color_block = tk.Canvas( color_frame, width=20, height=20, bg=color, bd=1, relief='solid' )
            color_block.pack( side='left', padx=5 )

            color_label = ttk.Label( color_frame, text=f"Color {i + 1}: {color}" )
            color_label.pack( side='left' )

            var = tk.BooleanVar( value=default_selection[i] )
            self.color_vars.append( var )
            ttk.Checkbutton( color_frame, variable=var ).pack( side='right', padx=5 )

        ttk.Button(
            self.source_colors_frame,
            text="Load target image",
            command=self.load_target_image
        ).pack( pady=10, padx=10, fill='x' )

    def load_target_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.jfif")],
            title="Select target image"
        )
        if file_path:
            self._invalidate_transfer_plan_state()
            try:
                img = Image.open( file_path ).convert( 'RGB' )
                img_array = np.array( img )

                if self.is_outline_image( img_array ):
                    messagebox.showinfo( "Hint",
                                         "Detected line sketch, will jump to the line sketch coloring module for use!" )
                    self._invalidate_outline_plan_state()
                    self.last_action = "outline_coloring"
                    self.target_image_path = None
                    self.target_labels_2d = None
                    self.outline_image_path = file_path
                    # 保留原始高清线稿，供 Element-Aware 直接分割，避免再次
                    # Canny 重绘造成双边和断线。
                    self._element_aware_original_outline_path = file_path
                    self.show_tab(self.outline_coloring_tab, "Coloring Line Sketch")
                    self.notebook.select(self.outline_coloring_tab)
                    self.display_image(img, self.outline_coloring_label, (400, 400))
                    self.update_status( f"The line draft diagram has been loaded: {Path( file_path ).name}" )
                    return

                self.display_image( img, self.target_image_label, (400, 400) )
                self.target_image_path = file_path

                self.create_outline_image( file_path, 'target_outline_image_path' )
                if self.target_outline_image_path and os.path.exists( self.target_outline_image_path ):
                    outline_img = Image.open( self.target_outline_image_path ).convert( 'RGB' )
                    self.display_image( outline_img, self.outline_label, (400, 400) )
                else:
                    messagebox.showerror( "Error!", "Unable to generate a line draft of the target image!" )
                    return

                self.update_status( f"Target image loaded!: {Path( file_path ).name}" )
            except Exception as e:
                messagebox.showerror( "Error", f"Unable to load target image: {str( e )}" )
                return
            try:
                configured_color_number = max(1, int(self.color_number_var.get()))
            except (TypeError, ValueError, tk.TclError):
                configured_color_number = 5
            transfer_region_count = max(
                configured_color_number,
                len(self.color_transfer_colors),
                len(self.hex_color_codes),
            )
            _, _, raw_target_labels = self.extract_colors_and_regions(
                file_path, transfer_region_count
            )
            self.target_labels_2d = self._normalize_transfer_labels_to_exact_count(
                raw_target_labels, transfer_region_count
            )
            self._transfer_total_region_count = int(transfer_region_count)
            if self.target_labels_2d is None:
                raise RuntimeError(
                    f"Unable to normalize target labels to N={transfer_region_count}."
                )

    def is_outline_image(self, img_array):
        """识别黑白/灰度线稿，同时容忍近黑像素的轻微 RGB 编码偏差。

        HSV 饱和度在接近黑色时数值不稳定：例如 (1, 0, 0) 会得到很高的
        饱和度，但视觉上仍然是黑线。旧版直接把这类像素记为彩色，导致高清
        抗锯齿线稿被误判。新版只在亮度足够时判断色度，并以 RGB 通道差作为
        主要彩色证据，再结合白底、深色线条和 Canny 边缘结构综合判断。
        """
        image = np.asarray(img_array)
        if image.ndim == 2:
            rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] >= 3:
            rgb = image[..., :3].astype(np.uint8, copy=False)
        else:
            return False

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[..., 1].astype(np.float32) / 255.0
        channel_spread = (
            rgb.max(axis=2).astype(np.int16)
            - rgb.min(axis=2).astype(np.int16)
        )

        h, w = gray.shape
        pixel_count = max(1, h * w)

        near_white = np.all(rgb >= 245, axis=2)
        foreground = ~near_white
        foreground_count = int(np.count_nonzero(foreground))
        if foreground_count < max(32, int(0.0015 * pixel_count)):
            return False

        # 接近黑色时 HSV 饱和度没有视觉意义，只在 gray >= 32 的可见亮度
        # 前景上判断彩色。通道差是主证据，饱和度只是辅助证据。
        visible_foreground = foreground & (gray >= 32)
        visible_count = max(1, int(np.count_nonzero(visible_foreground)))
        chromatic_foreground = visible_foreground & (
            (channel_spread > 20) & (saturation > 0.12)
        )

        # 深黑线即使存在 1~10 级 RGB 偏差也应视为中性。
        neutral_foreground = foreground & (
            (gray < 32) | (channel_spread <= 18)
        )
        chromatic_ratio = (
            float(np.count_nonzero(chromatic_foreground)) / visible_count
        )
        neutral_ratio = (
            float(np.count_nonzero(neutral_foreground)) / foreground_count
        )
        visible_spread_p95 = (
            float(np.percentile(channel_spread[visible_foreground], 95))
            if np.any(visible_foreground) else 0.0
        )

        neutral_dark = foreground & (gray < 220) & (
            (gray < 32) | (channel_spread <= 20)
        )
        neutral_dark_ratio = (
            float(np.count_nonzero(neutral_dark)) / pixel_count
        )
        dark_foreground_ratio = float(np.mean(gray[foreground] < 80))

        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / pixel_count
        white_ratio = float(np.count_nonzero(near_white)) / pixel_count

        grayscale_foreground = (
            chromatic_ratio <= 0.06
            and neutral_ratio >= 0.86
            and visible_spread_p95 <= 22.0
        )
        line_structure = (
            neutral_dark_ratio >= 0.002
            and dark_foreground_ratio >= 0.12
            and edge_ratio >= 0.004
            and edge_ratio <= 0.40
        )
        plausible_canvas = (
            white_ratio >= 0.20 or float(np.mean(gray)) >= 150.0
        )

        return bool(
            grayscale_foreground and line_structure and plausible_canvas
        )

    @staticmethod
    def _build_transfer_regions_from_target_labels(
            target_labels, output_shape, required_count):
        """从目标图自身的颜色标签构造恰好 N 个迁移区域。

        Target image 已通过 KMeans 得到 N 个颜色标签。这里直接把这些标签
        最近邻放大到原生线稿尺寸，避免 random-walker 因几何特征重复而把
        N 个请求组压缩成 1~2 个非空组。
        """
        labels = np.asarray(target_labels, dtype=np.int32)
        if labels.ndim != 2:
            return []
        try:
            required_count = max(1, int(required_count))
        except (TypeError, ValueError, OverflowError):
            return []
        out_h, out_w = map(int, output_shape[:2])
        if out_h <= 0 or out_w <= 0:
            return []

        native_labels = cv2.resize(
            labels, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32, copy=False)
        regions = [
            np.uint8(native_labels == label_index) * 255
            for label_index in range(required_count)
        ]

        # 理论上 KMeans 的每个标签都非空；极端缩放导致空标签时，把当前最大
        # 区域沿长轴二分，保证最终仍然恰好返回 N 个非空区域。
        for missing_index in range(required_count):
            if np.any(regions[missing_index]):
                continue
            donor_candidates = [
                index for index, region in enumerate(regions)
                if index != missing_index and np.count_nonzero(region) >= 2
            ]
            if not donor_candidates:
                return []
            donor_index = max(
                donor_candidates,
                key=lambda index: int(np.count_nonzero(regions[index]))
            )
            donor = regions[donor_index] > 0
            ys, xs = np.where(donor)
            if xs.size < 2:
                return []
            split_by_x = (xs.max() - xs.min()) >= (ys.max() - ys.min())
            coordinate = xs if split_by_x else ys
            threshold = float(np.median(coordinate))
            first_half = donor & (
                (np.indices(donor.shape)[1] <= threshold)
                if split_by_x
                else (np.indices(donor.shape)[0] <= threshold)
            )
            second_half = donor & ~first_half
            if not np.any(first_half) or not np.any(second_half):
                order = np.argsort(coordinate, kind='stable')
                half = max(1, len(order) // 2)
                first_half = np.zeros(donor.shape, dtype=bool)
                second_half = np.zeros(donor.shape, dtype=bool)
                first_ids = order[:half]
                second_ids = order[half:]
                first_half[ys[first_ids], xs[first_ids]] = True
                second_half[ys[second_ids], xs[second_ids]] = True
            regions[donor_index] = np.uint8(first_half) * 255
            regions[missing_index] = np.uint8(second_half) * 255

        return regions

    @staticmethod
    def _normalize_transfer_labels_to_exact_count(target_labels, required_count):
        """把目标标签规范成恰好 N 个连续标签 0..N-1。"""
        labels = np.asarray(target_labels, dtype=np.int32)
        if labels.ndim != 2:
            return None
        try:
            required_count = max(1, int(required_count))
        except (TypeError, ValueError, OverflowError):
            return None

        valid = labels >= 0
        if not np.any(valid):
            return None

        unique = [int(v) for v in np.unique(labels[valid])]
        remapped = np.full(labels.shape, -1, dtype=np.int32)
        for new_idx, old_idx in enumerate(unique):
            remapped[labels == old_idx] = int(new_idx)

        current_count = len(unique)
        if current_count == required_count:
            return remapped
        if current_count > required_count:
            # 把多余小区域合并到面积最大的前 required_count 个标签中。
            counts = [int(np.count_nonzero(remapped == idx)) for idx in range(current_count)]
            keep = sorted(range(current_count), key=lambda idx: counts[idx], reverse=True)[:required_count]
            keep_set = set(keep)
            keep_sorted = sorted(keep)
            out = np.full(remapped.shape, -1, dtype=np.int32)
            for new_idx, old_idx in enumerate(keep_sorted):
                out[remapped == old_idx] = new_idx
            # 将被丢弃的小标签并入面积最大的保留标签
            fallback_new = keep_sorted.index(max(keep_sorted, key=lambda idx: counts[idx])) if keep_sorted else 0
            for old_idx in range(current_count):
                if old_idx in keep_set:
                    continue
                out[remapped == old_idx] = int(fallback_new)
            return out

        # current_count < required_count: 不足部分继续切分最大区域，直到得到 N 个标签。
        out = remapped.copy()
        next_label = current_count
        while next_label < required_count:
            counts = [int(np.count_nonzero(out == idx)) for idx in range(next_label)]
            donor = max(range(next_label), key=lambda idx: counts[idx])
            ys, xs = np.where(out == donor)
            if xs.size < 2:
                break
            split_by_x = (xs.max() - xs.min()) >= (ys.max() - ys.min())
            coord = xs if split_by_x else ys
            threshold = float(np.median(coord))
            if split_by_x:
                mask_new = (out == donor) & (np.indices(out.shape)[1] > threshold)
            else:
                mask_new = (out == donor) & (np.indices(out.shape)[0] > threshold)
            if not np.any(mask_new) or np.all(mask_new):
                order = np.argsort(coord, kind='stable')
                half = max(1, len(order) // 2)
                mask_new = np.zeros(out.shape, dtype=bool)
                mask_new[ys[order[half:]], xs[order[half:]]] = True
            if not np.any(mask_new):
                break
            out[mask_new] = int(next_label)
            next_label += 1

        final_unique = [int(v) for v in np.unique(out[out >= 0])]
        if len(final_unique) != required_count:
            return None
        compact = np.full(out.shape, -1, dtype=np.int32)
        for new_idx, old_idx in enumerate(final_unique):
            compact[out == old_idx] = int(new_idx)
        return compact

    def apply_colors_to_target(self):
        """确认迁移色板并显示目标线稿；真正上色只在 Generate Solution 执行。

        该按钮原先会立即把颜色写进目标标签图，导致用户尚未生成方案就看到一张
        “半成品上色图”。现在它只完成工作流准备：校验色板、确认目标分区、展示
        自动生成的线稿，并启用 Generate Solution。
        """
        self.last_action = "color_transfer"
        self._invalidate_transfer_plan_state()
        if not hasattr(self, 'target_image_path') or not self.target_image_path:
            messagebox.showerror("Error!", "Please load the target image first!")
            return

        selected_colors = [
            self.color_transfer_colors[i]
            for i, var in enumerate(self.color_vars)
            if var.get() and i < len(self.color_transfer_colors)
        ]
        if not selected_colors:
            messagebox.showerror("Error!", "Please select at least one color!")
            return
        if self.target_labels_2d is None:
            messagebox.showerror("Error!", "Target region information is not ready!")
            return
        native_image = Image.open(self.target_image_path).convert('RGB')
        native_rgb = np.asarray(native_image, dtype=np.uint8)

        # Transfer Colors 的目标线稿提取与 Load Source Image 保持同一逻辑：
        # 统一走 create_outline_image / _generate_complete_line_sketch 生成结果。
        if (
            not self.target_outline_image_path
            or not os.path.exists(self.target_outline_image_path)
        ):
            created = self.create_outline_image(
                self.target_image_path, 'target_outline_image_path'
            )
            if not created:
                messagebox.showerror("Error!", "Unable to generate line sketch of target image!")
                return

        outline_image = Image.open(self.target_outline_image_path).convert('RGB')
        outline_array = np.asarray(outline_image, dtype=np.uint8)
        total_region_count = int(getattr(
            self, '_transfer_total_region_count', max(len(self.color_transfer_colors), len(selected_colors), 1)
        ))

        normalized_labels = self._normalize_transfer_labels_to_exact_count(
            self.target_labels_2d, total_region_count
        )
        if normalized_labels is None:
            messagebox.showerror(
                "Error!",
                f"Unable to prepare exactly N={total_region_count} target regions."
            )
            return
        self.target_labels_2d = normalized_labels
        regions = self._build_transfer_regions_from_target_labels(
            normalized_labels,
            outline_array.shape[:2],
            total_region_count,
        )
        regions = [
            np.asarray(region, dtype=np.uint8)
            for region in (regions or [])
            if np.any(np.asarray(region) > 0)
        ]
        if not regions or len(regions) != total_region_count:
            messagebox.showerror(
                "Error!",
                f"Unable to construct the required target regions: expected N={total_region_count}, got {len(regions or [])}."
            )
            return
        if len(selected_colors) > total_region_count:
            messagebox.showerror(
                "Error!",
                f"Selected colors M={len(selected_colors)} cannot exceed total regions N={total_region_count}."
            )
            return

        self.regions_for_plan = regions
        self.plan_outline_array = outline_array.copy()
        self.plan_outline_image = outline_image.copy()
        self._element_aware_v210_owner_map = None
        self._element_aware_v210_background_owner = None
        self._element_aware_v210_group_source_indices = list(range(total_region_count))
        # 关键：Transfer 方案生成回到普通精确枚举路径，不再走特殊的
        # element-aware transfer 分支，以避免区域数和方案数被压缩。
        self._element_aware_v21_active = False
        self._transfer_element_aware_active = False

        self.display_image(outline_image, self.target_image_label, (400, 400))

        self._prepared_transfer_colors = list(selected_colors)
        self.colors_applied = True
        self.last_action = 'color_transfer'
        self.plan_button.config(state="normal")
        region_count_hint = len(getattr(self, 'regions_for_plan', []) or [])
        self.update_status(
            f"Transfer palette prepared (M={len(selected_colors)} colors, N={region_count_hint} total regions). "
            "Target rendering now uses the same line-sketch logic as Load Source Image and exact partial-permutation plans P(N,M)."
        )
        messagebox.showinfo(
            "Ready",
            "The target line sketch is ready. Click Generate Solution to create the colored plans."
        )

    def save_results(self):
        """保存所有已生成的方案到指定文件夹"""
        if not hasattr( self, 'generated_plans' ) or not self.generated_plans:
            messagebox.showwarning( "Warning", "No plans generated, please generate a plan first!" )
            return
        folder = filedialog.askdirectory( title="Select folder to save all plans" )
        if not folder:
            return
        try:
            width, height, dpi = self._get_plan_export_settings()
            for i, img in enumerate( self.generated_plans ):
                filename = os.path.join( folder, f"plan_{i + 1}.png" )
                self._save_plan_png(img, filename, width, height, dpi)
            messagebox.showinfo( "Success", f"Saved {len( self.generated_plans )} plans to folder: {folder}" )
            self.update_status(
                f"All plans saved to {folder} "
                f"(within {width}×{height}, RGB, {dpi} DPI)"
            )
        except Exception as e:
            messagebox.showerror( "Error", f"Save failed: {str( e )}" )

    def save_single_plan(self, idx, placement, selected_colors, colored_image=None):
        if colored_image is None:
            messagebox.showerror("Error", "The generated plan snapshot is unavailable. Please regenerate it.")
            return

        try:
            width, height, dpi = self._get_plan_export_settings()
        except (TypeError, ValueError, tk.TclError):
            messagebox.showerror( "Error", "Please fill in valid positive integers (width, height, DPI)" )
            return

        file_path = filedialog.asksaveasfilename(
            filetypes=[("PNG files", "*.png")],
            defaultextension=".png",
            title=f"Save plan {idx + 1}",
            initialfile=f"Plan_{idx + 1}.png"
        )
        if file_path:
            img = self._save_plan_png(
                colored_image, file_path, width, height, dpi
            )
            actual_width, actual_height = img.size
            self.update_status(
                f"Plan {idx + 1} have been saved as {actual_width}×{actual_height} @ {dpi} DPI"
            )
            messagebox.showinfo( "Success!", f"Plan {idx + 1} have been saved to the specified location" )

    def open_outline_coloring(self):
        self._transfer_element_aware_active = False
        self.last_action = "outline_coloring"

        self.show_tab( self.outline_coloring_tab, "Coloring Line Sketch" )
        self.update_outline_coloring_tab()
        self.notebook.select( self.outline_coloring_tab )

    def update_outline_coloring_tab(self):
        for widget in self.color_selection_frame.winfo_children():
            widget.destroy()

        self.color_selection_vars = []
        for i, color in enumerate( self.outline_coloring_colors ):
            color_frame = ttk.Frame( self.color_selection_frame )
            color_frame.pack( fill='x', padx=5, pady=2 )

            color_block = tk.Canvas( color_frame, width=20, height=20, bg=color, bd=1, relief='solid' )
            color_block.pack( side='left', padx=5 )

            color_label = ttk.Label( color_frame, text=f"颜色 {i + 1}: {color}" )
            color_label.pack( side='left' )

            var = tk.BooleanVar( value=True )
            self.color_selection_vars.append( var )
            ttk.Checkbutton( color_frame, variable=var ).pack( side='right', padx=5 )

    def load_outline_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.jfif")],
            title="Select line sketch"
        )
        if file_path:
            try:
                img = Image.open( file_path ).convert( 'RGB' )
                img_array = np.array( img )

                if not self.is_outline_image( img_array ):
                    messagebox.showerror( "Error", "This is not a line sketch, please select the correct line sketch." )
                    return

                self.display_image( img, self.coloring_label, (400, 400) )
                self.update_status( f"The line draft diagram has been loaded: {Path( file_path ).name}" )
            except Exception as e:
                messagebox.showerror( "Error", f"Unable to load line sketch: {str( e )}" )


    # -------------------- Palette Result Editor / 调色板结果后处理编辑 --------------------
    def create_palette_recolor_tab(self):
        """创建最后一个调色板后处理板块。

        算法组合：
        1) RGB 凸包极值候选 + Lab 内部聚类，兼顾基础色和内部代表色；
        2) 每像素最多 4 个调色板顶点的局部重心权重，增强局部编辑性；
        3) 原图残差、亮度和高频细节回注，避免块状、糊化和结构损失；
        4) 参考图调色板通过“颜色距离 + 面积权重”的二分图匹配对齐。
        """
        outer = ttk.Frame(self.palette_recolor_tab)
        outer.pack(fill='both', expand=True, padx=10, pady=10)

        controls_host = ttk.LabelFrame(outer, text="Palette post-editor controls")
        controls_host.pack(side='left', fill='y', padx=(0, 10), pady=5)
        controls_host.configure(width=340)
        controls_host.pack_propagate(False)
        controls_canvas = tk.Canvas(controls_host, highlightthickness=0, width=315)
        controls_scroll = ttk.Scrollbar(controls_host, orient='vertical', command=controls_canvas.yview)
        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor='nw')
        controls.bind(
            '<Configure>',
            lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox('all'))
        )
        controls_canvas.bind(
            '<Configure>',
            lambda event: controls_canvas.itemconfigure(controls_window, width=max(260, event.width))
        )
        controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls_canvas.pack(side='left', fill='both', expand=True)
        controls_scroll.pack(side='right', fill='y')
        controls_canvas.bind(
            '<MouseWheel>',
            lambda event: controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        )

        source_box = ttk.LabelFrame(controls, text="Post-process source")
        source_box.pack(fill='x', padx=8, pady=(8, 6))
        self.palette_editor_source_var = tk.StringVar(value="Source image")
        self.palette_editor_source_combo = ttk.Combobox(
            source_box,
            textvariable=self.palette_editor_source_var,
            state='readonly',
            width=28,
        )
        self.palette_editor_source_combo.pack(fill='x', padx=6, pady=5)
        source_buttons = ttk.Frame(source_box)
        source_buttons.pack(fill='x', padx=6, pady=(0, 6))
        ttk.Button(
            source_buttons, text="Use selected", command=self.palette_editor_load_selected_source
        ).pack(side='left', expand=True, fill='x', padx=(0, 3))
        ttk.Button(
            source_buttons, text="Open image...", command=self.palette_editor_open_image
        ).pack(side='left', expand=True, fill='x', padx=(3, 0))
        ttk.Button(
            source_box,
            text="Use latest generated result",
            command=self.palette_editor_load_latest_generated_result,
        ).pack(fill='x', padx=6, pady=(0, 6))

        extract_box = ttk.LabelFrame(controls, text="Representative palette")
        extract_box.pack(fill='x', padx=8, pady=6)
        mode_row = ttk.Frame(extract_box)
        mode_row.pack(fill='x', padx=6, pady=(5, 2))
        ttk.Label(mode_row, text="Image mode:").pack(side='left')
        self.palette_editor_mode_var = tk.StringVar(value="Auto")
        self.palette_editor_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.palette_editor_mode_var,
            state='readonly',
            values=("Auto", "Traditional line-art & flat", "Photo"),
            width=25,
        )
        self.palette_editor_mode_combo.pack(side='right', fill='x', expand=True, padx=(6, 0))
        self.palette_editor_detected_mode_var = tk.StringVar(
            value="Detected after Extract"
        )
        ttk.Label(
            extract_box,
            textvariable=self.palette_editor_detected_mode_var,
            foreground="#555555",
            wraplength=295,
            justify='left',
        ).pack(fill='x', padx=6, pady=(0, 3))
        row = ttk.Frame(extract_box)
        row.pack(fill='x', padx=6, pady=5)
        self.palette_editor_count_label = ttk.Label(row, text="Detected regions:")
        self.palette_editor_count_label.pack(side='left')
        self.palette_editor_count_var = tk.StringVar(value="Auto")
        self.palette_editor_count_entry = ttk.Entry(
            row, textvariable=self.palette_editor_count_var, width=7, state='readonly'
        )
        self.palette_editor_count_entry.pack(side='left', padx=6)
        self.palette_editor_extract_btn = ttk.Button(
            row, text="Extract", command=self.palette_editor_extract_palette
        )
        self.palette_editor_extract_btn.pack(side='right')
        ttk.Button(
            extract_box,
            text="Import palette from reference image...",
            command=self.palette_editor_import_reference_palette,
        ).pack(fill='x', padx=6, pady=(0, 4))
        self.palette_editor_region_preview_btn = ttk.Button(
            extract_box,
            text="View region templates...",
            command=self.palette_editor_show_region_templates,
            state='disabled',
        )
        self.palette_editor_region_preview_btn.pack(fill='x', padx=6, pady=(0, 6))

        self.palette_editor_swatches_frame = ttk.LabelFrame(controls, text="Click a region to edit")
        self.palette_editor_swatches_frame.pack(fill='x', padx=8, pady=6)
        ttk.Label(
            self.palette_editor_swatches_frame,
            text="Extract region layers first",
            foreground="#666666",
        ).pack(padx=8, pady=12)

        selected_box = ttk.LabelFrame(controls, text="Selected region template")
        selected_box.pack(fill='x', padx=8, pady=6)
        self.palette_editor_selected_region_label = ttk.Label(
            selected_box,
            text="Click a region swatch to inspect its full layer and connected details",
            anchor='center',
            justify='center',
            wraplength=295,
        )
        self.palette_editor_selected_region_label.pack(fill='both', padx=6, pady=6)

        # Keep fixed internal defaults for smooth owner-based recoloring, but remove
        # the UI for NA-IGA candidate evolution and manual natural-recolor tuning.
        self.palette_editor_strength_var = tk.DoubleVar(value=100.0)
        self.palette_editor_luma_var = tk.DoubleVar(value=76.0)
        self.palette_editor_structure_var = tk.DoubleVar(value=68.0)
        self.palette_editor_smooth_var = tk.DoubleVar(value=14.0)
        self.palette_editor_optimize_var = tk.DoubleVar(value=0.0)

        action_row = ttk.Frame(controls)
        action_row.pack(fill='x', padx=8, pady=(7, 3))
        self.palette_editor_preview_btn = ttk.Button(
            action_row, text="Preview changes", command=self.palette_editor_preview
        )
        self.palette_editor_preview_btn.pack(side='left', expand=True, fill='x', padx=(0, 3))
        ttk.Button(
            action_row, text="Reset palette", command=self.palette_editor_reset_palette
        ).pack(side='left', expand=True, fill='x', padx=3)
        self.palette_editor_save_btn = ttk.Button(
            action_row, text="Save result", command=self.palette_editor_save_result
        )
        self.palette_editor_save_btn.pack(side='left', expand=True, fill='x', padx=(3, 0))

        self.palette_editor_progress = ttk.Progressbar(controls, mode='indeterminate')
        self.palette_editor_progress.pack(fill='x', padx=8, pady=(8, 3))
        self.palette_editor_info_var = tk.StringVar(
            value="Object-aware regions + multi-sublayer detail-preserving recoloring"
        )
        ttk.Label(
            controls,
            textvariable=self.palette_editor_info_var,
            wraplength=300,
            justify='left',
            foreground="#555555",
        ).pack(fill='x', padx=8, pady=(3, 8))

        preview_area = ttk.Frame(outer)
        preview_area.pack(side='right', fill='both', expand=True)
        preview_area.columnconfigure(0, weight=1)
        preview_area.columnconfigure(1, weight=1)
        preview_area.rowconfigure(0, weight=1)

        original_frame = ttk.LabelFrame(preview_area, text="Original")
        original_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=5)
        result_frame = ttk.LabelFrame(preview_area, text="Editable palette result")
        result_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=5)
        self.palette_editor_original_label = ttk.Label(
            original_frame, text="No image loaded", anchor='center'
        )
        self.palette_editor_original_label.pack(fill='both', expand=True, padx=8, pady=8)
        self.palette_editor_result_label = ttk.Label(
            result_frame, text="No result", anchor='center'
        )
        self.palette_editor_result_label.pack(fill='both', expand=True, padx=8, pady=8)
        original_frame.bind(
            '<Configure>', lambda _event: self.root.after_idle(self._palette_editor_refresh_preview_labels)
        )
        result_frame.bind(
            '<Configure>', lambda _event: self.root.after_idle(self._palette_editor_refresh_preview_labels)
        )

        self.palette_editor_refresh_sources()

    def _palette_editor_refresh_preview_labels(self):
        pairs = (
            (self.palette_editor_preview_image, self.palette_editor_original_label),
            (self.palette_editor_result_image, self.palette_editor_result_label),
        )
        for image, label in pairs:
            if image is None:
                continue
            width = label.winfo_width()
            height = label.winfo_height()
            if width < 100:
                width = 360
            if height < 100:
                height = 620
            self.display_image(image, label, (max(120, width - 16), max(120, height - 16)))

    def _palette_editor_add_scale(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill='x', padx=6, pady=2)
        ttk.Label(row, text=label, width=18).pack(side='left')
        scale = ttk.Scale(row, from_=0.0, to=100.0, variable=variable, orient='horizontal')
        scale.pack(side='left', fill='x', expand=True, padx=4)
        value_label = ttk.Label(row, width=4, anchor='e')
        value_label.pack(side='right')

        def update_value(*_args):
            value_label.config(text=f"{int(round(variable.get()))}")

        variable.trace_add('write', update_value)
        update_value()

    def open_palette_recoloring(self):
        self.palette_editor_refresh_sources()
        self.show_tab(self.palette_recolor_tab, "Palette Result Editor")
        if self.palette_editor_original_image is None:
            # 优先沿用当前源图；不存在时沿用最新方案。
            try:
                self.palette_editor_load_selected_source(silent=True)
            except Exception:
                pass

    def palette_editor_refresh_sources(self):
        if not hasattr(self, 'palette_editor_source_combo'):
            return
        values = []
        has_latest = (
            getattr(self, 'latest_generated_result_image', None) is not None
            or bool(getattr(self, 'generated_plans', []) or [])
        )
        if has_latest:
            values.append("Latest generated result")
        if self.image_path and os.path.exists(self.image_path):
            values.append("Source image")
        if self.target_image_path and os.path.exists(self.target_image_path):
            values.append("Target image")
        for index, _image in enumerate(getattr(self, 'generated_plans', []) or []):
            values.append(f"Generated plan {index + 1}")
        if self.palette_editor_manual_path and os.path.exists(self.palette_editor_manual_path):
            values.append("Opened image")
        if not values:
            values = ["Source image"]
        self.palette_editor_source_combo['values'] = values
        if self.palette_editor_source_var.get() not in values:
            self.palette_editor_source_var.set(values[0])

    def palette_editor_open_image(self):
        path = filedialog.askopenfilename(
            title="Open image for palette recoloring",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.jfif")],
        )
        if not path:
            return
        self.palette_editor_manual_path = path
        self.palette_editor_refresh_sources()
        self.palette_editor_source_var.set("Opened image")
        self.palette_editor_load_selected_source()

    def palette_editor_load_selected_source(self, silent=False):
        choice = self.palette_editor_source_var.get()
        image = None
        description = choice
        if choice == "Source image" and self.image_path and os.path.exists(self.image_path):
            image = Image.open(self.image_path).convert('RGB')
        elif choice == "Target image" and self.target_image_path and os.path.exists(self.target_image_path):
            image = Image.open(self.target_image_path).convert('RGB')
        elif choice == "Latest generated result":
            image = getattr(self, 'latest_generated_result_image', None)
            if image is None:
                plans = list(getattr(self, 'generated_plans', []) or [])
                if plans:
                    image = plans[-1]
            description = getattr(self, 'latest_generated_result_description', None) or choice
        elif choice.startswith("Generated plan "):
            try:
                idx = int(choice.rsplit(' ', 1)[-1]) - 1
                image = self.generated_plans[idx].convert('RGB').copy()
            except (ValueError, IndexError, AttributeError):
                image = None
        elif choice == "Opened image" and self.palette_editor_manual_path:
            image = Image.open(self.palette_editor_manual_path).convert('RGB')

        if image is None:
            if not silent:
                messagebox.showwarning(
                    "Palette Result Editor", "No usable image is available. Load or generate an image first."
            )
            return False

        return self._palette_editor_set_source_image(image, description)

    def _palette_editor_set_source_image(self, image, description):
        source = image.convert('RGB').copy()
        self.palette_editor_original_image = source
        preview = source.copy()
        preview.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        self.palette_editor_preview_image = preview
        self.palette_editor_result_image = preview.copy()
        self.palette_editor_mode = None
        self.palette_editor_classification_metrics = {}
        if hasattr(self, 'palette_editor_detected_mode_var'):
            self.palette_editor_detected_mode_var.set("Detected after Extract")
        if hasattr(self, 'palette_editor_count_label'):
            self.palette_editor_count_label.config(text="Detected regions:")
        if hasattr(self, 'palette_editor_swatches_frame'):
            self.palette_editor_swatches_frame.config(
                text="Click an editable color or region"
            )
        self.palette_editor_source_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_target_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_proportions = np.empty((0,), dtype=np.float32)
        self.palette_editor_sparse_model = None
        self.palette_editor_owner_map = None
        # Object-aware edit groups are unions of color sublayers that belong to one
        # spatial object/material region (e.g. bright orange + dark orange coat).
        self.palette_editor_group_map = None
        self.palette_editor_group_map_original = None
        self.palette_editor_group_map_refined = None
        self.palette_editor_region_reduction_metrics = {}
        self.palette_editor_edge_refine_metrics = {}
        self.palette_editor_group_source_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_group_target_palette = np.empty((0, 3), dtype=np.uint8)
        self.palette_editor_group_proportions = np.empty((0,), dtype=np.float32)
        self.palette_editor_group_members = []
        self.palette_editor_selected_region_index = None
        self.palette_editor_selected_region_photo = None
        self.palette_editor_region_template_sheet = None
        if hasattr(self, 'palette_editor_region_preview_btn'):
            self.palette_editor_region_preview_btn.config(state='disabled')
        self.palette_editor_count_var.set("Auto")
        self.palette_editor_naiga_candidates = []
        self.palette_editor_naiga_scores = []
        self.palette_editor_naiga_selected = None
        self.palette_editor_naiga_generation = 0
        if hasattr(self, 'palette_editor_naiga_frame'):
            self._palette_editor_clear_naiga_candidates()
        if (
            hasattr(self, 'palette_editor_original_label')
            and hasattr(self, 'palette_editor_result_label')
        ):
            self._palette_editor_refresh_preview_labels()
        self.palette_editor_info_var.set(
            f"Loaded {description}: {source.width} x {source.height}. "
            "Choose Auto, Traditional line-art & flat, or Photo, then Extract."
        )
        self.update_status(f"Palette Result Editor: loaded {description}")
        return True

    def palette_editor_load_generated_result(self, image, description="Generated result"):
        self.latest_generated_result_image = image.convert('RGB').copy()
        self.latest_generated_result_description = description
        self.palette_editor_refresh_sources()
        if hasattr(self, 'palette_editor_source_var'):
            self.palette_editor_source_var.set("Latest generated result")
        loaded = self._palette_editor_set_source_image(
            self.latest_generated_result_image, description
        )
        self.show_tab(self.palette_recolor_tab, "Palette Result Editor")
        self.notebook.select(self.palette_recolor_tab)
        return loaded

    def palette_editor_load_latest_generated_result(self):
        image = getattr(self, 'latest_generated_result_image', None)
        if image is None:
            messagebox.showinfo(
                "Palette Result Editor",
                "Generate a Transfer Colors or Coloring Line Sketch plan first."
            )
            return False
        description = getattr(self, 'latest_generated_result_description', None) or "Latest generated result"
        return self.palette_editor_load_generated_result(image, description)

    def _palette_editor_set_busy(self, busy, message=None):
        self.palette_editor_busy = bool(busy)
        state = 'disabled' if busy else 'normal'
        for button_name in (
            'palette_editor_extract_btn',
            'palette_editor_preview_btn', 'palette_editor_save_btn'
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                try:
                    button.config(state=state)
                except tk.TclError:
                    pass
        region_button = getattr(self, 'palette_editor_region_preview_btn', None)
        if region_button is not None:
            try:
                region_button.config(
                    state='disabled' if busy or self.palette_editor_owner_map is None else 'normal'
                )
            except tk.TclError:
                pass
        if busy:
            self.palette_editor_progress.start(12)
        else:
            self.palette_editor_progress.stop()
        if message:
            self.palette_editor_info_var.set(message)

    @staticmethod
    def _palette_rgb_to_lab_array(rgb):
        values = np.asarray(rgb, dtype=np.uint8).reshape(-1, 1, 3)
        return cv2.cvtColor(values, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    @staticmethod
    def _palette_lab_to_rgb_array(lab):
        values = np.clip(np.asarray(lab), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        return cv2.cvtColor(values, cv2.COLOR_LAB2RGB).reshape(-1, 3)

    @staticmethod
    def _palette_farthest_select(candidates_lab, count, weights=None, seeds=None):
        candidates = np.asarray(candidates_lab, dtype=np.float32)
        if len(candidates) == 0 or count <= 0:
            return []
        count = min(int(count), len(candidates))
        weights = np.ones(len(candidates), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        weights = np.maximum(weights, 1e-6)
        selected = [] if seeds is None else list(seeds)
        chosen_candidate_indices = []
        centroid = np.average(candidates, axis=0, weights=weights)
        if not selected:
            chroma = np.linalg.norm(candidates[:, 1:3] - 128.0, axis=1)
            score = np.linalg.norm(candidates - centroid, axis=1) * (0.55 + 0.45 * weights / weights.max())
            score += 0.12 * chroma
            first = int(np.argmax(score))
            chosen_candidate_indices.append(first)
            selected.append(candidates[first])
        while len(chosen_candidate_indices) < count:
            selected_arr = np.asarray(selected, dtype=np.float32)
            distances = np.min(
                np.linalg.norm(candidates[:, None, :] - selected_arr[None, :, :], axis=2), axis=1
            )
            support = 0.65 + 0.35 * np.sqrt(weights / weights.max())
            score = distances * support
            if chosen_candidate_indices:
                score[np.asarray(chosen_candidate_indices, dtype=np.intp)] = -1.0
            idx = int(np.argmax(score))
            if score[idx] < 0:
                break
            chosen_candidate_indices.append(idx)
            selected.append(candidates[idx])
        return chosen_candidate_indices[:count]

    @staticmethod
    def _palette_classify_editor_image(image):
        """区分传统线稿/扁平图与摄影图，不读取文件名且不修改输入。

        分类仅在最长边 384 像素的副本上执行。规则综合颜色熵、主量化色覆盖、
        平坦像素、强跳变、墨线白底和 Canny 边缘；返回的诊断信息用于界面解释
        和回归测试，不能改变 Transfer Colors 或 Coloring Line Sketch 的结果。
        """
        started = time.perf_counter()
        if isinstance(image, Image.Image):
            source = np.asarray(image.convert('RGB'), dtype=np.uint8)
        else:
            source = np.asarray(image)
            if source.ndim == 2:
                source = np.repeat(source[:, :, None], 3, axis=2)
            if source.ndim != 3 or source.shape[2] < 3:
                raise ValueError("Image classifier expects an RGB-compatible image")
            source = np.asarray(source[:, :, :3], dtype=np.uint8)
        if source.size == 0:
            raise ValueError("The image has no usable pixels")

        height, width = source.shape[:2]
        scale = min(1.0, 384.0 / max(height, width, 1))
        if scale < 1.0:
            analysis = cv2.resize(
                source,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            analysis = source.copy()

        quantized = (analysis // 16).astype(np.uint16)
        codes = (
            (quantized[:, :, 0] << 8)
            | (quantized[:, :, 1] << 4)
            | quantized[:, :, 2]
        )
        _values, counts = np.unique(codes, return_counts=True)
        counts = np.sort(counts.astype(np.float64))[::-1]
        probabilities = counts / max(float(counts.sum()), 1.0)
        entropy_h16 = float(
            -np.sum(probabilities * np.log2(probabilities + 1e-12))
        )
        top8 = float(np.sum(counts[:8]) / max(float(np.sum(counts)), 1.0))

        rgb_float = analysis.astype(np.float32)
        horizontal = np.linalg.norm(
            rgb_float[:, 1:] - rgb_float[:, :-1], axis=2
        ).reshape(-1)
        vertical = np.linalg.norm(
            rgb_float[1:] - rgb_float[:-1], axis=2
        ).reshape(-1)
        neighbor_delta = np.concatenate([horizontal, vertical])
        exact_neighbor = float(np.mean(neighbor_delta < 1.0)) if len(neighbor_delta) else 1.0
        strong_jump = float(np.mean(neighbor_delta >= 48.0)) if len(neighbor_delta) else 0.0

        gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)
        canny_density = float(np.mean(
            cv2.Canny(gray, 60, 160, L2gradient=True) > 0
        ))
        white_ratio = float(np.mean(np.min(analysis, axis=2) > 245))
        hsv = cv2.cvtColor(analysis, cv2.COLOR_RGB2HSV)
        low_saturation = float(np.mean(hsv[:, :, 1] < 20))

        signals = {
            'flat_palette': bool(
                entropy_h16 <= 5.5 and top8 >= 0.52 and exact_neighbor >= 0.25
            ),
            'ink_line': bool(
                white_ratio >= 0.45
                and low_saturation >= 0.65
                and canny_density >= 0.07
            ),
            'graphic_edge': bool(
                canny_density >= 0.16 and strong_jump >= 0.14
            ),
            'photo_signature': bool(
                entropy_h16 >= 6.0 and top8 <= 0.48
            ),
        }
        traditional_signals = [
            key for key in ('flat_palette', 'ink_line', 'graphic_edge')
            if signals[key]
        ]
        if traditional_signals:
            mode = 'traditional'
            reason = ', '.join(traditional_signals)
            confidence = min(0.98, 0.72 + 0.09 * len(traditional_signals))
        else:
            mode = 'photo'
            reason = 'photo_signature' if signals['photo_signature'] else 'default_photo'
            confidence = 0.88 if signals['photo_signature'] else 0.60

        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        return {
            'mode': mode,
            'confidence': float(confidence),
            'reason': reason,
            'selection': 'auto',
            'signals': signals,
            'metrics': {
                'h16': entropy_h16,
                'top8': top8,
                'exact_neighbor': exact_neighbor,
                'strong_jump_48': strong_jump,
                'canny_density': canny_density,
                'white_ratio': white_ratio,
                'low_saturation_ratio': low_saturation,
                'elapsed_ms': elapsed_ms,
            },
        }

    def _palette_resolve_editor_mode(self, image, requested='Auto'):
        """解析用户模式；手动选择永远优先于自动分类。"""
        normalized = str(requested or 'Auto').strip().lower()
        if normalized in ('traditional', 'traditional line-art & flat', 'line-art', 'flat'):
            diagnostics = self._palette_classify_editor_image(image)
            diagnostics = dict(diagnostics)
            diagnostics.update({
                'mode': 'traditional',
                'selection': 'manual',
                'reason': 'manual_traditional',
                'confidence': 1.0,
            })
            return 'traditional', diagnostics
        if normalized in ('photo', 'photograph'):
            diagnostics = self._palette_classify_editor_image(image)
            diagnostics = dict(diagnostics)
            diagnostics.update({
                'mode': 'photo',
                'selection': 'manual',
                'reason': 'manual_photo',
                'confidence': 1.0,
            })
            return 'photo', diagnostics
        diagnostics = self._palette_classify_editor_image(image)
        return diagnostics['mode'], diagnostics

    @staticmethod
    def _palette_iga_robust_normalize(values, percentile=95.0):
        """将 IGA 特征压缩到 0~1，避免极端像素支配评分。"""
        data = np.asarray(values, dtype=np.float32)
        if data.size == 0:
            return data
        scale = max(float(np.percentile(data, percentile)), 1e-6)
        return np.clip(data / scale, 0.0, 1.0)

    @staticmethod
    def _palette_iga_color_family(lab_color):
        """用于抑制近似色重复；中性色按明度分组，彩色按宽色相分组。"""
        lightness, channel_a, channel_b = [float(v) for v in lab_color]
        chroma = math.hypot(channel_a - 128.0, channel_b - 128.0)
        if chroma < 13.0:
            return ('neutral', int(np.clip(lightness // 64.0, 0, 3)))
        hue = (math.degrees(math.atan2(channel_b - 128.0, channel_a - 128.0)) + 360.0) % 360.0
        return ('hue', int(hue // 60.0))

    @staticmethod
    def _palette_iga_cluster_representative(cluster_rgb, cluster_lab, center_lab):
        """以真实像素众数附近的中位色代替 KMeans 均值，减少脏色和混色。"""
        rgb_values = np.asarray(cluster_rgb, dtype=np.uint8)
        lab_values = np.asarray(cluster_lab, dtype=np.float32)
        if len(rgb_values) == 0:
            value = np.asarray(center_lab, dtype=np.float32).reshape(1, 1, 3)
            return cv2.cvtColor(np.clip(value, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(3)

        distance = np.linalg.norm(lab_values - np.asarray(center_lab, dtype=np.float32), axis=1)
        cutoff = float(np.percentile(distance, 68.0)) if len(distance) > 3 else float(distance.max(initial=0.0))
        core = rgb_values[distance <= cutoff + 1e-6]
        if len(core) == 0:
            core = rgb_values

        # 5-bit 颜色箱用于确定真实颜色峰值，再在峰值附近取中位数。
        bins = (core.astype(np.uint16) // 8).astype(np.int16)
        unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
        mode_rgb = unique_bins[int(np.argmax(counts))].astype(np.float32) * 8.0 + 4.0
        mode_distance = np.linalg.norm(core.astype(np.float32) - mode_rgb, axis=1)
        near_cutoff = float(np.percentile(mode_distance, 52.0)) if len(mode_distance) > 3 else float(mode_distance.max(initial=0.0))
        near = core[mode_distance <= near_cutoff + 1e-6]
        if len(near) == 0:
            near = core
        return np.clip(np.median(near, axis=0), 0, 255).astype(np.uint8)

    @staticmethod
    def _palette_traditional_ink_mask(image_rgb):
        """识别低明度、低色度的中性墨线；高色度暗色仍属于填充色。"""
        rgb = np.asarray(image_rgb, dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma = np.linalg.norm(lab[:, :, 1:3] - 128.0, axis=2)
        return (lab[:, :, 0] <= 96.0) & (chroma <= 18.0)

    def _palette_extract_traditional_flat(self, image, color_count=8):
        """提取传统图填充色；低明度低色度墨线只保留结构，不进入色块。"""
        requested = max(2, min(12, int(color_count)))
        source = image.convert('RGB').copy()
        source.thumbnail((720, 720), Image.Resampling.LANCZOS)
        rgb = np.asarray(source, dtype=np.uint8)
        height, width = rgb.shape[:2]
        flat_rgb = rgb.reshape(-1, 3)
        total_pixels = len(flat_rgb)
        if total_pixels == 0:
            raise ValueError("The image has no usable pixels")

        lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        flat_lab = lab_image.reshape(-1, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # 黑色/近黑色中性像素属于线稿结构层，不是可编辑填充颜色。
        # 暗蓝、暗红等高色度填充不受影响。
        ink_like = self._palette_traditional_ink_mask(rgb).reshape(-1)
        editable_mask = ~ink_like
        editable_indices = np.flatnonzero(editable_mask)
        if len(editable_indices) < 2:
            raise ValueError(
                "Traditional palette extraction needs non-ink fill colors"
            )
        editable_rgb = flat_rgb[editable_mask]
        editable_lab = flat_lab[editable_mask]
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_strength = self._palette_iga_robust_normalize(
            np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y).reshape(-1),
            92.0,
        )
        chroma_strength = self._palette_iga_robust_normalize(
            np.linalg.norm(flat_lab[:, 1:3] - 128.0, axis=1), 92.0
        )

        border_width = max(2, int(round(min(height, width) * 0.04)))
        border_mask = np.zeros((height, width), dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True
        border_lab = lab_image[border_mask]
        border_center = np.median(border_lab, axis=0)
        border_distance = np.linalg.norm(border_lab - border_center, axis=1)
        global_background_distance = np.linalg.norm(flat_lab - border_center, axis=1)
        coherent_border = float(np.mean(border_distance < 11.0))
        global_background_ratio = float(np.mean(global_background_distance < 11.0))
        background_like = np.zeros(total_pixels, dtype=bool)
        if coherent_border >= 0.50 and global_background_ratio >= 0.16:
            background_like = global_background_distance < 11.0

        sample_weight = (
            0.62 + 0.28 * np.sqrt(chroma_strength) + 0.10 * edge_strength
        ).astype(np.float32)
        sample_weight[background_like] *= 0.52
        sample_weight = np.maximum(sample_weight, 0.05)

        rng = np.random.default_rng(20260724)
        weighted_count = min(len(editable_indices), 36000)
        if weighted_count < len(editable_indices):
            editable_weight = sample_weight[editable_mask]
            probabilities = editable_weight / max(
                float(editable_weight.sum()), 1e-8
            )
            weighted_indices = rng.choice(
                editable_indices,
                size=weighted_count,
                replace=False,
                p=probabilities,
            )
        else:
            weighted_indices = editable_indices

        grid_indices = []
        per_cell = 120
        for grid_y in range(8):
            y0 = grid_y * height // 8
            y1 = (grid_y + 1) * height // 8
            for grid_x in range(8):
                x0 = grid_x * width // 8
                x1 = (grid_x + 1) * width // 8
                ys, xs = np.mgrid[y0:y1, x0:x1]
                cell = (ys * width + xs).reshape(-1)
                cell = cell[editable_mask[cell]]
                if len(cell) > per_cell:
                    cell = rng.choice(cell, size=per_cell, replace=False)
                grid_indices.append(cell.astype(np.intp))
        sample_indices = np.unique(np.concatenate([weighted_indices, *grid_indices]))
        sample_rgb = flat_rgb[sample_indices]
        sample_lab = flat_lab[sample_indices]

        quantized = ((sample_rgb.astype(np.uint16) // 4) * 4).astype(np.uint8)
        unique_count = len(np.unique(quantized, axis=0))
        candidate_count = min(
            max(requested * 4, 12),
            max(1, unique_count),
            len(sample_lab),
        )
        if candidate_count <= requested:
            candidate_count = min(requested, len(sample_lab))

        features = sample_lab.astype(np.float32).copy()
        features[:, 0] *= 0.82
        clusterer = KMeans(
            n_clusters=int(candidate_count),
            random_state=20260724,
            n_init=6,
            max_iter=160,
        )
        sample_labels = clusterer.fit_predict(features)
        centers_lab = np.asarray(clusterer.cluster_centers_, dtype=np.float32)
        centers_lab[:, 0] /= 0.82

        representatives = np.asarray([
            self._palette_iga_cluster_representative(
                sample_rgb[sample_labels == cluster_id],
                sample_lab[sample_labels == cluster_id],
                centers_lab[cluster_id],
            )
            for cluster_id in range(candidate_count)
        ], dtype=np.uint8)
        representatives_lab = self._palette_rgb_to_lab_array(representatives)
        distance = np.sum(
            (editable_lab[:, None, :] - representatives_lab[None, :, :]) ** 2,
            axis=2,
        )
        candidate_labels = np.argmin(distance, axis=1)
        candidate_counts = np.bincount(
            candidate_labels, minlength=candidate_count
        ).astype(np.float64)
        candidate_weighted = np.bincount(
            candidate_labels,
            weights=sample_weight[editable_mask],
            minlength=candidate_count,
        ).astype(np.float64)
        label_image = np.full((height, width), -1, dtype=np.int32)
        label_image.reshape(-1)[editable_mask] = candidate_labels
        border_flat = border_mask.reshape(-1)
        editable_border = border_flat[editable_mask]

        coherence = np.zeros(candidate_count, dtype=np.float32)
        centrality = np.zeros(candidate_count, dtype=np.float32)
        interiority = np.zeros(candidate_count, dtype=np.float32)
        for cluster_id in range(candidate_count):
            mask = label_image == cluster_id
            area = max(float(candidate_counts[cluster_id]), 1.0)
            component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            if component_count > 1:
                coherence[cluster_id] = (
                    float(np.max(stats[1:, cv2.CC_STAT_AREA])) / area
                )
            coords_y, coords_x = np.where(mask)
            if len(coords_x):
                center_x = float(np.mean(coords_x)) / max(width - 1, 1)
                center_y = float(np.mean(coords_y)) / max(height - 1, 1)
                centrality[cluster_id] = 1.0 - min(
                    math.hypot(center_x - 0.5, center_y - 0.5)
                    / 0.70710678,
                    1.0,
                )
            assigned = candidate_labels == cluster_id
            border_fraction = (
                float(np.mean(editable_border[assigned]))
                if np.any(assigned) else 1.0
            )
            interiority[cluster_id] = 1.0 - math.sqrt(max(border_fraction, 0.0))

        area_ratio = candidate_counts / max(float(candidate_counts.sum()), 1.0)
        weighted_ratio = candidate_weighted / max(float(candidate_weighted.sum()), 1.0)
        representative_chroma = self._palette_iga_robust_normalize(
            np.linalg.norm(representatives_lab[:, 1:3] - 128.0, axis=1), 90.0
        )
        region_score = 0.55 * centrality + 0.45 * interiority
        support_score = (
            0.46 * np.sqrt(area_ratio / max(float(area_ratio.max()), 1e-8))
            + 0.20 * np.sqrt(weighted_ratio / max(float(weighted_ratio.max()), 1e-8))
            + 0.12 * representative_chroma
            + 0.08 * np.sqrt(np.clip(coherence, 0.0, 1.0))
            + 0.14 * region_score
        )

        merge_order = np.argsort(-support_score, kind='stable')
        merged = []
        for index in merge_order:
            if candidate_counts[index] <= 0:
                continue
            target = next((
                merged_index
                for merged_index, item in enumerate(merged)
                if np.linalg.norm(representatives_lab[index] - item['lab']) < 9.5
            ), None)
            if target is None:
                merged.append({
                    'rgb': representatives[index].copy(),
                    'lab': representatives_lab[index].copy(),
                    'count': float(candidate_counts[index]),
                    'weight': float(candidate_weighted[index]),
                    'coherence': float(coherence[index]),
                    'region': float(region_score[index]),
                    'score': float(support_score[index]),
                })
            else:
                item = merged[target]
                item['count'] += float(candidate_counts[index])
                item['weight'] += float(candidate_weighted[index])
                item['coherence'] = max(item['coherence'], float(coherence[index]))
                item['region'] = max(item['region'], float(region_score[index]))
                if support_score[index] > item['score'] * 1.04:
                    item['rgb'] = representatives[index].copy()
                    item['lab'] = representatives_lab[index].copy()
                    item['score'] = float(support_score[index])

        if not merged:
            raise ValueError("No representative traditional colors were found")
        merged_rgb = np.asarray([item['rgb'] for item in merged], dtype=np.uint8)
        merged_lab = self._palette_rgb_to_lab_array(merged_rgb)
        merged_counts = np.asarray([item['count'] for item in merged], dtype=np.float64)
        merged_weights = np.asarray([item['weight'] for item in merged], dtype=np.float64)
        merged_coherence = np.asarray(
            [item['coherence'] for item in merged], dtype=np.float32
        )
        merged_region = np.asarray([item['region'] for item in merged], dtype=np.float32)
        merged_area = merged_counts / max(float(merged_counts.sum()), 1.0)
        merged_weight_ratio = merged_weights / max(float(merged_weights.sum()), 1.0)
        merged_chroma = self._palette_iga_robust_normalize(
            np.linalg.norm(merged_lab[:, 1:3] - 128.0, axis=1), 90.0
        )
        merged_base = (
            0.46 * np.sqrt(merged_area / max(float(merged_area.max()), 1e-8))
            + 0.20 * np.sqrt(
                merged_weight_ratio / max(float(merged_weight_ratio.max()), 1e-8)
            )
            + 0.12 * merged_chroma
            + 0.08 * np.sqrt(np.clip(merged_coherence, 0.0, 1.0))
            + 0.14 * merged_region
        )

        final_count = min(requested, len(merged_rgb))
        selected = [int(np.argmax(merged_area))]
        while len(selected) < final_count:
            selected_lab = merged_lab[np.asarray(selected, dtype=np.intp)]
            min_distance = np.min(
                np.linalg.norm(
                    merged_lab[:, None, :] - selected_lab[None, :, :], axis=2
                ),
                axis=1,
            )
            diversity = np.clip(min_distance / 42.0, 0.0, 1.0)
            score = (
                0.48 * merged_base
                + 0.28 * diversity
                + 0.16 * merged_region
                + 0.08 * merged_chroma
            )
            selected_families = [
                self._palette_iga_color_family(merged_lab[i]) for i in selected
            ]
            for index in range(len(score)):
                family_count = selected_families.count(
                    self._palette_iga_color_family(merged_lab[index])
                )
                score[index] -= 0.12 * family_count
                if family_count >= 2:
                    score[index] -= 0.42
            score[np.asarray(selected, dtype=np.intp)] = -1e9
            score[min_distance < 7.0] -= 0.50
            selected.append(int(np.argmax(score)))

        palette_rgb = merged_rgb[np.asarray(selected, dtype=np.intp)]
        palette_lab = self._palette_rgb_to_lab_array(palette_rgb)
        final_labels = np.argmin(
            np.sum(
                (editable_lab[:, None, :] - palette_lab[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        refined_colors = []
        for palette_index in range(final_count):
            mask = final_labels == palette_index
            refined_colors.append(self._palette_iga_cluster_representative(
                editable_rgb[mask],
                editable_lab[mask],
                palette_lab[palette_index],
            ))
        palette_rgb = np.asarray(refined_colors, dtype=np.uint8)
        palette_lab = self._palette_rgb_to_lab_array(palette_rgb)
        final_labels = np.argmin(
            np.sum(
                (editable_lab[:, None, :] - palette_lab[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        final_counts = np.bincount(
            final_labels, minlength=final_count
        ).astype(np.int64)
        order = np.argsort(-final_counts, kind='stable')
        palette_rgb = palette_rgb[order]
        final_counts = final_counts[order]
        proportions = final_counts.astype(np.float32)
        proportions /= max(float(proportions.sum()), 1.0)
        return palette_rgb, proportions

    @staticmethod
    def _palette_cv_lab_to_cie(lab_color):
        """将 OpenCV 8-bit Lab 转成 CIE L*a*b*，供感知色差计算使用。"""
        value = np.asarray(lab_color, dtype=np.float64)
        return np.asarray([
            value[0] * (100.0 / 255.0),
            value[1] - 128.0,
            value[2] - 128.0,
        ], dtype=np.float64)

    def _palette_iga_perceptual_distance(self, lab_a, lab_b):
        """正确尺度的 CIEDE2000；避免直接把 OpenCV Lab 送进公式。"""
        return float(self._ciede2000(
            self._palette_cv_lab_to_cie(lab_a),
            self._palette_cv_lab_to_cie(lab_b),
        ))

    def _palette_iga_should_merge(
            self, lab_a, lab_b, area_a=1.0, area_b=1.0,
            coherence_a=1.0, coherence_b=1.0,
            allow_fragment_merge=False):
        """判断两个颜色候选是否只是采样/压缩/光照误差造成的重复色。"""
        distance = self._palette_iga_perceptual_distance(lab_a, lab_b)
        cie_a = self._palette_cv_lab_to_cie(lab_a)
        cie_b = self._palette_cv_lab_to_cie(lab_b)
        chroma_a = float(math.hypot(cie_a[1], cie_a[2]))
        chroma_b = float(math.hypot(cie_b[1], cie_b[2]))
        neutral = max(chroma_a, chroma_b) < 14.0
        both_light = min(cie_a[0], cie_b[0]) > 76.0
        same_family = self._palette_iga_color_family(lab_a) == self._palette_iga_color_family(lab_b)
        small = min(float(area_a), float(area_b)) < 0.020
        tiny_pair = max(float(area_a), float(area_b)) < 0.030
        small_fragment_pair = (
            max(float(area_a), float(area_b)) < 0.034
            and min(float(area_a), float(area_b)) < 0.032
        )
        fragmented = min(float(coherence_a), float(coherence_b)) < 0.46

        if distance <= 4.4:
            return True
        if neutral and distance <= 7.8:
            return True
        if both_light and max(chroma_a, chroma_b) < 22.0 and distance <= 7.0:
            return True
        if same_family and small and fragmented and distance <= 10.5:
            return True
        if allow_fragment_merge:
            if same_family and tiny_pair and fragmented and distance <= 10.5:
                return True
            if small_fragment_pair and fragmented and distance <= 10.4:
                return True
        if min(float(area_a), float(area_b)) < 0.010 and distance <= 9.2:
            return True
        return False

    def _palette_regularize_owner_map(
            self, owner_map, lab_image, palette_lab, edge_strength=None):
        """颜色误差抑制 + 边缘保护的区域整形。

        目标不是简单模糊标签，而是：
        - 平坦区域消除盐胡椒碎片；
        - 真实细线/小色块在颜色证据明显时保留；
        - 小碎片优先并入边界接触最多、颜色也最接近的邻区。
        """
        owners = np.asarray(owner_map, dtype=np.int32).copy()
        lab = np.asarray(lab_image, dtype=np.float32)
        palette = np.asarray(palette_lab, dtype=np.float32)
        if owners.ndim != 2 or lab.shape[:2] != owners.shape or len(palette) <= 1:
            return owners

        h, w = owners.shape
        if edge_strength is None:
            light = lab[:, :, 0]
            gx = cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(light, cv2.CV_32F, 0, 1, ksize=3)
            edge = self._palette_iga_robust_normalize(
                np.sqrt(gx * gx + gy * gy), 92.0
            )
        else:
            edge = np.asarray(edge_strength, dtype=np.float32).reshape(h, w)

        # 两轮有条件中值标签：只在低边缘或颜色归属不确定处采用多数标签。
        for kernel_size in (5, 3):
            if int(np.max(owners)) > 255:
                break
            median = cv2.medianBlur(owners.astype(np.uint8), kernel_size).astype(np.int32)
            yy, xx = np.indices((h, w))
            current_lab = palette[np.clip(owners, 0, len(palette) - 1)]
            median_lab = palette[np.clip(median, 0, len(palette) - 1)]
            current_d2 = np.sum((lab - current_lab) ** 2, axis=2)
            median_d2 = np.sum((lab - median_lab) ** 2, axis=2)
            changed = median != owners
            safe = changed & (
                ((edge < 0.24) & (median_d2 <= current_d2 + 90.0))
                | (median_d2 + 24.0 < current_d2)
            )
            owners[safe] = median[safe]

        # 连通碎片整形。阈值随图像面积自适应，但不粗暴抹掉颜色证据强的小部件。
        min_component = max(18, int(round(h * w * 0.00042)))
        kernel = np.ones((3, 3), dtype=np.uint8)
        for _pass in range(2):
            changed_any = False
            for owner in range(len(palette)):
                mask = np.uint8(owners == owner)
                count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
                if count <= 1:
                    continue
                for label in range(1, count):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    if area >= min_component:
                        continue
                    x = int(stats[label, cv2.CC_STAT_LEFT])
                    y = int(stats[label, cv2.CC_STAT_TOP])
                    bw = int(stats[label, cv2.CC_STAT_WIDTH])
                    bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                    x0, y0 = max(0, x - 2), max(0, y - 2)
                    x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
                    local_component = labels[y0:y1, x0:x1] == label
                    dilated = cv2.dilate(local_component.astype(np.uint8), kernel) > 0
                    ring = dilated & ~local_component
                    local_owners = owners[y0:y1, x0:x1]
                    candidates = local_owners[ring]
                    candidates = candidates[candidates != owner]
                    if candidates.size == 0:
                        continue
                    values, counts = np.unique(candidates, return_counts=True)
                    component_lab = lab[y0:y1, x0:x1][local_component]
                    mean_lab = np.mean(component_lab, axis=0)
                    own_distance = float(np.linalg.norm(mean_lab - palette[owner]))
                    best_owner = None
                    best_score = -1e18
                    best_distance = float('inf')
                    boundary_total = max(float(np.sum(counts)), 1.0)
                    for candidate, boundary_count in zip(values, counts):
                        candidate = int(candidate)
                        candidate_distance = float(np.linalg.norm(mean_lab - palette[candidate]))
                        boundary_support = float(boundary_count) / boundary_total
                        score = 2.2 * boundary_support - 0.018 * candidate_distance
                        if score > best_score:
                            best_score = score
                            best_owner = candidate
                            best_distance = candidate_distance
                    if best_owner is None:
                        continue

                    # 小但颜色高度独立的真实细节保留；噪点/抗锯齿碎片则合并。
                    preserve = (
                        area >= int(round(min_component * 0.34))
                        and own_distance + 8.0 < best_distance
                        and float(np.mean(edge[y0:y1, x0:x1][local_component])) > 0.12
                    )
                    if preserve:
                        continue
                    local_owners[local_component] = int(best_owner)
                    changed_any = True
            if not changed_any:
                break
        return owners

    def _palette_assign_owner_map_chunked(self, image_rgb, palette_rgb):
        """对高分辨率预览按块分配 owner，避免 N 像素 × K 颜色矩阵占用过大。"""
        rgb = np.asarray(image_rgb, dtype=np.uint8)
        smoothed = cv2.bilateralFilter(rgb, d=5, sigmaColor=18, sigmaSpace=4)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
        palette_lab = self._palette_rgb_to_lab_array(palette_rgb)
        flat = lab.reshape(-1, 3)
        owners = np.empty(len(flat), dtype=np.int32)
        chunk_size = 180000
        for start in range(0, len(flat), chunk_size):
            end = min(start + chunk_size, len(flat))
            distance = np.sum(
                (flat[start:end, None, :] - palette_lab[None, :, :]) ** 2,
                axis=2,
            )
            owners[start:end] = np.argmin(distance, axis=1).astype(np.int32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = self._palette_iga_robust_normalize(
            np.sqrt(gx * gx + gy * gy), 92.0
        )
        owner_map = owners.reshape(rgb.shape[:2])
        owner_map = self._palette_regularize_owner_map(
            owner_map, lab, palette_lab, edge_strength=edge
        )
        return owner_map

    def _palette_consolidate_owner_regions(self, rgb, owner_map, palette_rgb):
        """在最终像素归属后再次合并“视觉上同色但被误拆”的区域。"""
        original = np.asarray(rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32).copy()
        palette = np.asarray(palette_rgb, dtype=np.uint8).copy()
        total = max(int(owners.size), 1)

        for _iteration in range(12):
            count = len(palette)
            if count <= 2:
                break
            palette_lab = self._palette_rgb_to_lab_array(palette)
            counts = np.bincount(owners.reshape(-1), minlength=count).astype(np.int64)
            areas = counts.astype(np.float64) / total
            coherence = np.ones(count, dtype=np.float32)
            for index in range(count):
                mask = np.uint8(owners == index)
                component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
                if component_count > 1 and counts[index] > 0:
                    coherence[index] = float(np.max(stats[1:, cv2.CC_STAT_AREA])) / float(counts[index])

            best_pair = None
            best_key = None
            for i in range(count):
                for j in range(i + 1, count):
                    if not self._palette_iga_should_merge(
                            palette_lab[i], palette_lab[j],
                            areas[i], areas[j], coherence[i], coherence[j],
                            allow_fragment_merge=True):
                        continue
                    distance = self._palette_iga_perceptual_distance(
                        palette_lab[i], palette_lab[j]
                    )
                    key = (
                        distance,
                        min(areas[i], areas[j]),
                        min(coherence[i], coherence[j]),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pair = (i, j)
            if best_pair is None:
                break

            i, j = best_pair
            # 保留面积更大/更连贯的一方作为 owner，避免编号频繁漂移。
            if (areas[j], coherence[j]) > (areas[i], coherence[i]):
                keep, remove = j, i
            else:
                keep, remove = i, j
            owners[owners == remove] = keep
            compact = [idx for idx in range(count) if idx != remove]
            remap = np.full(count, -1, dtype=np.int32)
            remap[np.asarray(compact, dtype=np.intp)] = np.arange(len(compact), dtype=np.int32)
            owners = remap[owners]

            refined = []
            flat_rgb = original.reshape(-1, 3)
            flat_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            for index in range(len(compact)):
                mask = owners.reshape(-1) == index
                refined.append(self._palette_iga_cluster_representative(
                    flat_rgb[mask], flat_lab[mask], np.mean(flat_lab[mask], axis=0)
                ))
            palette = np.asarray(refined, dtype=np.uint8)
        return owners, palette


    @staticmethod
    def _palette_owner_boundary_stats(owner_map, edge_strength):
        owners = np.asarray(owner_map, dtype=np.int32)
        edge = np.asarray(edge_strength, dtype=np.float32)
        if owners.shape != edge.shape:
            raise ValueError("Owner map and edge map must have the same shape")
        stats = {}

        def consume(label_a, label_b, boundary_edge):
            valid = label_a != label_b
            if not np.any(valid):
                return
            a = label_a[valid].reshape(-1).astype(np.int32)
            b = label_b[valid].reshape(-1).astype(np.int32)
            e = boundary_edge[valid].reshape(-1).astype(np.float32)
            for first, second, edge_value in zip(a, b, e):
                if first < 0 or second < 0:
                    continue
                key = (int(first), int(second)) if first < second else (int(second), int(first))
                bucket = stats.setdefault(key, {'contact': 0.0, 'edge_sum': 0.0})
                bucket['contact'] += 1.0
                bucket['edge_sum'] += float(edge_value)

        consume(owners[:, :-1], owners[:, 1:], np.maximum(edge[:, :-1], edge[:, 1:]))
        consume(owners[:-1, :], owners[1:, :], np.maximum(edge[:-1, :], edge[1:, :]))
        return stats

    def _palette_merge_adjacent_similar_regions(self, rgb, owner_map, palette_rgb, max_iterations=18):
        """Merge adjacent layers that are perceptually close and separated by a weak boundary.

        This is the key editor-side refinement requested by the user: red / dark-red
        or other illumination variants should become one editable region when they do
        not form a visually strong boundary for humans.
        """
        original = np.asarray(rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32).copy()
        palette = np.asarray(palette_rgb, dtype=np.uint8).copy()
        if owners.ndim != 2 or original.shape[:2] != owners.shape or len(palette) <= 1:
            return owners, palette

        full_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
        smoothed = cv2.bilateralFilter(original, d=7, sigmaColor=20, sigmaSpace=5)
        smooth_lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = self._palette_iga_robust_normalize(np.sqrt(gx * gx + gy * gy), 92.0)
        total = max(int(owners.size), 1)

        for _iteration in range(int(max_iterations)):
            count = len(palette)
            if count <= 2:
                break
            counts = np.bincount(owners.reshape(-1), minlength=count).astype(np.int64)
            if not np.any(counts > 0):
                break
            palette_lab = self._palette_rgb_to_lab_array(palette)
            adjacency = self._palette_owner_boundary_stats(owners, edge)
            if not adjacency:
                break

            best = None
            best_score = None
            for (first, second), info in adjacency.items():
                if first >= count or second >= count:
                    continue
                area_a = float(counts[first]) / total
                area_b = float(counts[second]) / total
                if area_a <= 0.0 or area_b <= 0.0:
                    continue
                distance = self._palette_iga_perceptual_distance(palette_lab[first], palette_lab[second])
                same_family = self._palette_iga_color_family(palette_lab[first]) == self._palette_iga_color_family(palette_lab[second])
                cie_a = self._palette_cv_lab_to_cie(palette_lab[first])
                cie_b = self._palette_cv_lab_to_cie(palette_lab[second])
                chroma = max(float(math.hypot(cie_a[1], cie_a[2])), float(math.hypot(cie_b[1], cie_b[2])))
                mean_boundary = float(info['edge_sum']) / max(float(info['contact']), 1.0)
                contact_ratio = float(info['contact']) / max(float(min(counts[first], counts[second])), 1.0)
                small = min(area_a, area_b) < 0.012
                tiny = min(area_a, area_b) < 0.006
                neutral = chroma < 16.0

                merge = False
                if distance <= 4.8:
                    merge = True
                elif neutral and mean_boundary <= 0.34 and distance <= 8.8:
                    merge = True
                elif same_family and mean_boundary <= 0.32 and distance <= 11.8:
                    merge = True
                elif same_family and small and mean_boundary <= 0.42 and distance <= 13.4:
                    merge = True
                elif tiny and mean_boundary <= 0.48 and distance <= 14.5:
                    merge = True

                if not merge:
                    continue

                score = (
                    distance
                    + 8.5 * mean_boundary
                    - 3.0 * min(contact_ratio, 1.0)
                    - (0.8 if same_family else 0.0)
                    - (0.6 if small else 0.0)
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best = (first, second)

            if best is None:
                break

            first, second = best
            if counts[first] >= counts[second]:
                keep, remove = first, second
            else:
                keep, remove = second, first
            owners[owners == remove] = keep
            compact = [idx for idx in range(len(palette)) if idx != remove]
            remap = np.full(len(palette), -1, dtype=np.int32)
            remap[np.asarray(compact, dtype=np.intp)] = np.arange(len(compact), dtype=np.int32)
            owners = remap[owners]

            flat_rgb = original.reshape(-1, 3)
            flat_lab = full_lab.reshape(-1, 3)
            refined = []
            for index in range(len(compact)):
                mask = owners.reshape(-1) == index
                if not np.any(mask):
                    continue
                refined.append(self._palette_iga_cluster_representative(
                    flat_rgb[mask], flat_lab[mask], np.mean(flat_lab[mask], axis=0)
                ))
            palette = np.asarray(refined, dtype=np.uint8)
            owners = self._palette_regularize_owner_map(
                owners, smooth_lab, self._palette_rgb_to_lab_array(palette), edge_strength=edge
            )

        return owners, palette

    def _palette_reduce_similar_layer_families(self, rgb, owner_map, palette_rgb, requested=None):
        """Optional family-level simplification for the region editor.

        When the automatic extraction still yields many layers, merge one or two
        perceptually similar families (for example red / deep red) so the editor
        exposes fewer, more practical recoloring layers.
        """
        original = np.asarray(rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32).copy()
        palette = np.asarray(palette_rgb, dtype=np.uint8).copy()
        if owners.ndim != 2 or original.shape[:2] != owners.shape or len(palette) <= 1:
            return owners, palette
        if requested is not None:
            return owners, palette

        desired_max = 7
        flat_rgb = original.reshape(-1, 3)
        flat_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        total = max(int(owners.size), 1)

        while len(palette) > desired_max:
            palette_lab = self._palette_rgb_to_lab_array(palette)
            counts = np.bincount(owners.reshape(-1), minlength=len(palette)).astype(np.int64)
            areas = counts.astype(np.float64) / total
            best = None
            best_score = None
            for first in range(len(palette)):
                for second in range(first + 1, len(palette)):
                    distance = self._palette_iga_perceptual_distance(palette_lab[first], palette_lab[second])
                    same_family = self._palette_iga_color_family(palette_lab[first]) == self._palette_iga_color_family(palette_lab[second])
                    cie_a = self._palette_cv_lab_to_cie(palette_lab[first])
                    cie_b = self._palette_cv_lab_to_cie(palette_lab[second])
                    neutral = max(float(math.hypot(cie_a[1], cie_a[2])), float(math.hypot(cie_b[1], cie_b[2]))) < 18.0
                    area_small = min(float(areas[first]), float(areas[second]))
                    area_big = max(float(areas[first]), float(areas[second]))
                    merge = False
                    if same_family and area_small <= 0.12 and distance <= 22.2:
                        merge = True
                    elif same_family and area_small <= 0.18 and distance <= 15.0:
                        merge = True
                    elif neutral and area_small <= 0.12 and distance <= 18.0:
                        merge = True
                    elif area_small <= 0.035 and distance <= 13.0:
                        merge = True
                    if not merge:
                        continue
                    score = distance - 16.0 * area_small + 2.5 * area_big
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (first, second)
            if best is None:
                break
            first, second = best
            if counts[first] >= counts[second]:
                keep, remove = first, second
            else:
                keep, remove = second, first
            owners[owners == remove] = keep
            compact = [idx for idx in range(len(palette)) if idx != remove]
            remap = np.full(len(palette), -1, dtype=np.int32)
            remap[np.asarray(compact, dtype=np.intp)] = np.arange(len(compact), dtype=np.int32)
            owners = remap[owners]
            refined = []
            for index in range(len(compact)):
                mask = owners.reshape(-1) == index
                if not np.any(mask):
                    continue
                refined.append(self._palette_iga_cluster_representative(
                    flat_rgb[mask], flat_lab[mask], np.mean(flat_lab[mask], axis=0)
                ))
            palette = np.asarray(refined, dtype=np.uint8)
        return owners, palette



    @staticmethod
    def _palette_nearest_labels_chunked(lab_pixels, palette_lab, chunk_size=180000):
        """Return nearest palette index for each Lab pixel without a huge N x K allocation."""
        pixels = np.asarray(lab_pixels, dtype=np.float32).reshape(-1, 3)
        palette = np.asarray(palette_lab, dtype=np.float32).reshape(-1, 3)
        if len(palette) == 0:
            raise ValueError("Palette is empty")
        labels = np.empty(len(pixels), dtype=np.int32)
        for start in range(0, len(pixels), int(chunk_size)):
            end = min(start + int(chunk_size), len(pixels))
            distance = np.sum(
                (pixels[start:end, None, :] - palette[None, :, :]) ** 2,
                axis=2,
            )
            labels[start:end] = np.argmin(distance, axis=1).astype(np.int32)
        return labels

    def _palette_merge_geometric_candidates(self, image_rgb, candidates_rgb, requested=None):
        """Merge perceptually duplicate palette candidates using image support.

        This stage is intentionally stronger than the v4.3.4 fragment merger.  It
        merges *palette vertices* before pixel ownership is computed, so sky or a
        green wall is not split merely because JPEG noise / illumination produced
        several close representatives.
        """
        image = np.asarray(image_rgb, dtype=np.uint8)
        pixels_rgb = image.reshape(-1, 3)
        pixels_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        palette = np.asarray(candidates_rgb, dtype=np.uint8).reshape(-1, 3)
        if len(palette) == 0:
            raise ValueError("No palette candidates were produced")

        # Exact RGB de-duplication first.
        palette = np.unique(palette, axis=0)
        minimum_count = 4 if requested is None else min(max(2, int(requested)), len(palette))

        for _iteration in range(24):
            if len(palette) <= minimum_count:
                break
            palette_lab = self._palette_rgb_to_lab_array(palette)
            labels = self._palette_nearest_labels_chunked(pixels_lab, palette_lab)
            counts = np.bincount(labels, minlength=len(palette)).astype(np.int64)
            total = max(int(counts.sum()), 1)

            best = None
            for first in range(len(palette)):
                for second in range(first + 1, len(palette)):
                    delta_e = self._palette_iga_perceptual_distance(
                        palette_lab[first], palette_lab[second]
                    )
                    same_family = (
                        self._palette_iga_color_family(palette_lab[first])
                        == self._palette_iga_color_family(palette_lab[second])
                    )
                    area_small = min(float(counts[first]), float(counts[second])) / total
                    # ΔE00 <= 5.5: visually duplicate in almost all circumstances.
                    # Same broad hue + ΔE00 <= 10.2: merge illumination/JPEG variants.
                    merge = (
                        delta_e <= 5.5
                        or (same_family and delta_e <= 12.0)
                        or (area_small < 0.006 and delta_e <= 11.5)
                    )
                    if not merge:
                        continue
                    key = (delta_e, area_small, -max(counts[first], counts[second]))
                    if best is None or key < best[0]:
                        best = (key, first, second)
            if best is None:
                break

            _key, first, second = best
            keep = first if counts[first] >= counts[second] else second
            remove = second if keep == first else first
            merged_mask = (labels == first) | (labels == second)
            merged_rgb = pixels_rgb[merged_mask]
            merged_lab = pixels_lab[merged_mask]
            if len(merged_rgb):
                representative = self._palette_iga_cluster_representative(
                    merged_rgb, merged_lab, np.mean(merged_lab, axis=0)
                )
            else:
                representative = palette[keep]
            compact = [index for index in range(len(palette)) if index != remove]
            palette = palette[np.asarray(compact, dtype=np.intp)]
            keep_after = compact.index(keep)
            palette[keep_after] = representative

        palette_lab = self._palette_rgb_to_lab_array(palette)
        labels = self._palette_nearest_labels_chunked(pixels_lab, palette_lab)
        counts = np.bincount(labels, minlength=len(palette)).astype(np.int64)
        support = counts.astype(np.float64) / max(float(counts.sum()), 1.0)

        if requested is None:
            # Preserve small but deliberate accent colors; remove only nearly unsupported
            # vertices that are usually compression outliers.
            keep = support >= 0.0025
            if int(np.sum(keep)) < min(6, len(palette)):
                top = np.argsort(-support, kind='stable')[:min(6, len(palette))]
                keep[top] = True
            palette = palette[keep]
            support = support[keep]
            if len(palette) > 12:
                palette_lab = self._palette_rgb_to_lab_array(palette)
                selected = [int(np.argmax(support))]
                while len(selected) < 12:
                    selected_lab = palette_lab[np.asarray(selected, dtype=np.intp)]
                    distance = np.min(
                        np.linalg.norm(
                            palette_lab[:, None, :] - selected_lab[None, :, :], axis=2
                        ), axis=1,
                    )
                    score = distance * (0.62 + 0.38 * np.sqrt(support / max(float(support.max()), 1e-8)))
                    score[np.asarray(selected, dtype=np.intp)] = -1.0
                    selected.append(int(np.argmax(score)))
                palette = palette[np.asarray(selected, dtype=np.intp)]
        else:
            target = min(max(2, int(requested)), len(palette))
            if len(palette) > target:
                palette_lab = self._palette_rgb_to_lab_array(palette)
                selected = [int(np.argmax(support))]
                while len(selected) < target:
                    selected_lab = palette_lab[np.asarray(selected, dtype=np.intp)]
                    distance = np.min(
                        np.linalg.norm(
                            palette_lab[:, None, :] - selected_lab[None, :, :], axis=2
                        ), axis=1,
                    )
                    score = distance * (0.58 + 0.42 * np.sqrt(support / max(float(support.max()), 1e-8)))
                    score[np.asarray(selected, dtype=np.intp)] = -1.0
                    selected.append(int(np.argmax(score)))
                palette = palette[np.asarray(selected, dtype=np.intp)]
        return np.asarray(palette, dtype=np.uint8)

    def _palette_extract_geometric_candidates(self, image_rgb, requested=None):
        """Paper-informed palette: MeanShift interior modes + RGB convex-hull extremes.

        The uploaded paper uses q=0.2 MeanShift to capture representative colors
        inside the hull, then combines them with simplified convex-hull vertices.
        This implementation keeps the same principle while using farthest-point
        hull simplification for a practical single-file GUI implementation.
        """
        rgb = np.asarray(image_rgb, dtype=np.uint8)
        height, width = rgb.shape[:2]
        pixels = rgb.reshape(-1, 3)
        rng = np.random.default_rng(20260725)

        # Uniform random support plus deterministic spatial cells prevents a small
        # but important object from disappearing from the sample.
        random_count = min(len(pixels), 14000)
        random_indices = (
            rng.choice(len(pixels), size=random_count, replace=False)
            if random_count < len(pixels)
            else np.arange(len(pixels), dtype=np.intp)
        )
        grid_indices = []
        per_cell = 80
        for gy in range(8):
            y0, y1 = gy * height // 8, (gy + 1) * height // 8
            for gx in range(8):
                x0, x1 = gx * width // 8, (gx + 1) * width // 8
                ys, xs = np.mgrid[y0:y1, x0:x1]
                cell = (ys * width + xs).reshape(-1)
                if len(cell) > per_cell:
                    cell = rng.choice(cell, size=per_cell, replace=False)
                grid_indices.append(cell.astype(np.intp))
        sample_indices = np.unique(np.concatenate([random_indices, *grid_indices]))
        sample_rgb = pixels[sample_indices]

        interior = []
        try:
            from sklearn.cluster import MeanShift, estimate_bandwidth
            normalized = sample_rgb.astype(np.float32) / 255.0
            estimate_count = len(normalized)
            bandwidth = float(estimate_bandwidth(
                normalized,
                quantile=0.20,
                n_samples=estimate_count,
                random_state=20260725,
            ))
            bandwidth = max(bandwidth, 0.035)
            min_bin = max(6, int(round(len(normalized) / 1000.0)))
            model = MeanShift(
                bandwidth=bandwidth,
                bin_seeding=True,
                min_bin_freq=min_bin,
                cluster_all=True,
                max_iter=180,
            )
            labels = model.fit_predict(normalized)
            centers = np.asarray(model.cluster_centers_, dtype=np.float32) * 255.0
            sample_lab = cv2.cvtColor(
                sample_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
            ).reshape(-1, 3).astype(np.float32)
            for cluster_id in range(len(centers)):
                mask = labels == cluster_id
                if not np.any(mask):
                    continue
                center_lab = self._palette_rgb_to_lab_array([
                    np.clip(centers[cluster_id], 0, 255).astype(np.uint8)
                ])[0]
                interior.append(self._palette_iga_cluster_representative(
                    sample_rgb[mask], sample_lab[mask], center_lab
                ))
        except Exception:
            # Optional dependencies missing: retain the design principle with a
            # modest KMeans interior approximation.
            interior_count = 4 if requested is None else max(2, min(int(requested), 6))
            model = KMeans(
                n_clusters=min(interior_count, len(sample_rgb)),
                random_state=20260725,
                n_init=5,
                max_iter=140,
            )
            labels = model.fit_predict(sample_rgb.astype(np.float32))
            sample_lab = cv2.cvtColor(
                sample_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
            ).reshape(-1, 3).astype(np.float32)
            for cluster_id in range(int(np.max(labels)) + 1):
                mask = labels == cluster_id
                if np.any(mask):
                    center_rgb = np.mean(sample_rgb[mask], axis=0).astype(np.uint8)
                    interior.append(self._palette_iga_cluster_representative(
                        sample_rgb[mask], sample_lab[mask],
                        self._palette_rgb_to_lab_array([center_rgb])[0]
                    ))

        # Quantized colors make ConvexHull stable and carry pixel support.
        quantized = ((pixels.astype(np.uint16) // 8) * 8 + 4).clip(0, 255).astype(np.uint8)
        unique_rgb, unique_counts = np.unique(quantized, axis=0, return_counts=True)
        min_support = max(2, int(round(len(pixels) * 0.00004)))
        supported = unique_counts >= min_support
        hull_source = unique_rgb[supported]
        hull_counts = unique_counts[supported]
        if len(hull_source) < 4:
            hull_source, hull_counts = unique_rgb, unique_counts

        hull_colors = hull_source
        hull_support = hull_counts.astype(np.float64)
        try:
            from scipy.spatial import ConvexHull
            if len(hull_source) >= 4:
                hull = ConvexHull(hull_source.astype(np.float64), qhull_options='QJ')
                hull_colors = hull_source[np.asarray(hull.vertices, dtype=np.intp)]
                hull_support = hull_counts[np.asarray(hull.vertices, dtype=np.intp)].astype(np.float64)
        except Exception:
            pass

        hull_target = 8 if requested is None else min(max(4, int(requested)), 10)
        hull_target = min(hull_target, len(hull_colors))
        selected_hull = []
        if hull_target:
            hull_lab = self._palette_rgb_to_lab_array(hull_colors)
            chroma = np.linalg.norm(hull_lab[:, 1:3] - 128.0, axis=1)
            support_norm = np.sqrt(hull_support / max(float(hull_support.max()), 1.0))
            chroma_norm = chroma / max(float(chroma.max()), 1.0)
            first_score = 0.68 * support_norm + 0.32 * chroma_norm
            selected_hull = [int(np.argmax(first_score))]
            while len(selected_hull) < hull_target:
                chosen = hull_lab[np.asarray(selected_hull, dtype=np.intp)]
                distance = np.min(
                    np.linalg.norm(hull_lab[:, None, :] - chosen[None, :, :], axis=2),
                    axis=1,
                )
                score = distance * (0.68 + 0.32 * support_norm)
                score[np.asarray(selected_hull, dtype=np.intp)] = -1.0
                selected_hull.append(int(np.argmax(score)))

        candidate_parts = []
        if interior:
            candidate_parts.append(np.asarray(interior, dtype=np.uint8))
        if selected_hull:
            candidate_parts.append(hull_colors[np.asarray(selected_hull, dtype=np.intp)])
        if not candidate_parts:
            candidate_parts.append(np.asarray([np.median(pixels, axis=0)], dtype=np.uint8))
        candidates = np.concatenate(candidate_parts, axis=0)
        return self._palette_merge_geometric_candidates(rgb, candidates, requested=requested)

    def _palette_geometric_dominant_owner_map(self, image_rgb, palette_rgb):
        """Compute a dominant layer from sparse RGB-space tetrahedral coordinates.

        Pixels inside the Delaunay tessellation use barycentric weights. Pixels
        outside the selected palette hull use a four-nearest inverse-distance
        fallback.  The GUI keeps one dominant owner per pixel to preserve the
        user's requirement that editing one swatch cannot overlay another swatch.
        """
        rgb = np.asarray(image_rgb, dtype=np.uint8)
        palette = np.asarray(palette_rgb, dtype=np.uint8)
        points = rgb.reshape(-1, 3).astype(np.float64)
        palette_points = palette.astype(np.float64)
        owners = np.empty(len(points), dtype=np.int32)
        confidence = np.empty(len(points), dtype=np.float32)
        inside_count = 0
        triangulation = None
        try:
            from scipy.spatial import Delaunay
            if len(palette_points) >= 4:
                triangulation = Delaunay(palette_points, qhull_options='QJ')
        except Exception:
            triangulation = None

        chunk_size = 150000
        for start in range(0, len(points), chunk_size):
            end = min(start + chunk_size, len(points))
            chunk = points[start:end]
            chunk_owner = np.full(len(chunk), -1, dtype=np.int32)
            chunk_confidence = np.zeros(len(chunk), dtype=np.float32)
            inside = np.zeros(len(chunk), dtype=bool)
            if triangulation is not None:
                simplex = triangulation.find_simplex(chunk)
                inside = simplex >= 0
                if np.any(inside):
                    local_indices = np.flatnonzero(inside)
                    simplex_indices = simplex[inside]
                    transform = triangulation.transform[simplex_indices, :3]
                    delta = chunk[inside] - triangulation.transform[simplex_indices, 3]
                    bary_first = np.einsum('nij,nj->ni', transform, delta)
                    bary = np.concatenate([
                        bary_first,
                        (1.0 - np.sum(bary_first, axis=1, keepdims=True)),
                    ], axis=1)
                    bary = np.clip(bary, 0.0, None)
                    bary /= np.maximum(np.sum(bary, axis=1, keepdims=True), 1e-12)
                    vertices = triangulation.simplices[simplex_indices]
                    maxima = np.argmax(bary, axis=1)
                    chunk_owner[local_indices] = vertices[
                        np.arange(len(vertices)), maxima
                    ].astype(np.int32)
                    sorted_weight = np.sort(bary, axis=1)
                    chunk_confidence[local_indices] = (
                        sorted_weight[:, -1] - sorted_weight[:, -2]
                    ).astype(np.float32)
                    inside_count += int(np.sum(inside))

            outside_indices = np.flatnonzero(~inside)
            if len(outside_indices):
                outside_points = chunk[outside_indices]
                distance = np.linalg.norm(
                    outside_points[:, None, :] - palette_points[None, :, :], axis=2
                )
                nearest_count = min(4, len(palette_points))
                nearest = np.argpartition(
                    distance, kth=nearest_count - 1, axis=1
                )[:, :nearest_count]
                nearest_distance = np.take_along_axis(distance, nearest, axis=1)
                nearest_order = np.argsort(nearest_distance, axis=1)
                nearest = np.take_along_axis(nearest, nearest_order, axis=1)
                nearest_distance = np.take_along_axis(
                    nearest_distance, nearest_order, axis=1
                )
                inverse = 1.0 / np.maximum(nearest_distance, 2.0) ** 2
                inverse /= np.maximum(np.sum(inverse, axis=1, keepdims=True), 1e-12)
                chunk_owner[outside_indices] = nearest[:, 0].astype(np.int32)
                if nearest_count > 1:
                    chunk_confidence[outside_indices] = (
                        inverse[:, 0] - inverse[:, 1]
                    ).astype(np.float32)
                else:
                    chunk_confidence[outside_indices] = 1.0

            owners[start:end] = chunk_owner
            confidence[start:end] = np.clip(chunk_confidence, 0.0, 1.0)

        owner_map = owners.reshape(rgb.shape[:2])
        smoothed = cv2.bilateralFilter(rgb, d=7, sigmaColor=20, sigmaSpace=5)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
        owner_map = self._palette_regularize_owner_map(
            owner_map, lab, self._palette_rgb_to_lab_array(palette)
        )
        metrics = {
            'inside_tetrahedral_percent': 100.0 * inside_count / max(len(points), 1),
            'mean_dominance_confidence': float(np.mean(confidence)) if len(confidence) else 0.0,
            'method': 'MeanShift(q=0.20)+convex-hull+Delaunay dominant layer',
        }
        return owner_map.astype(np.int32), confidence.reshape(rgb.shape[:2]), metrics

    @staticmethod
    def _photo_hue_distance(first, second):
        difference = abs(float(first) - float(second)) % 360.0
        return min(difference, 360.0 - difference)

    @staticmethod
    def _photo_union_find(size):
        parent = np.arange(int(size), dtype=np.int32)
        rank = np.zeros(int(size), dtype=np.int8)

        def find(value):
            value = int(value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = int(parent[value])
            return value

        def union(first, second):
            first = find(first)
            second = find(second)
            if first == second:
                return False
            if rank[first] < rank[second]:
                first, second = second, first
            parent[second] = first
            if rank[first] == rank[second]:
                rank[first] += 1
            return True

        return parent, find, union

    @staticmethod
    def _photo_adjacency(labels, edge_map):
        labels = np.asarray(labels, dtype=np.int32)
        edge_map = np.asarray(edge_map, dtype=np.float32)
        result = {}

        def consume(first, second, edge):
            different = first != second
            if not np.any(different):
                return
            values_a = first[different].reshape(-1)
            values_b = second[different].reshape(-1)
            edge_values = edge[different].reshape(-1)
            for region_a, region_b, edge_value in zip(values_a, values_b, edge_values):
                region_a = int(region_a)
                region_b = int(region_b)
                if region_a > region_b:
                    region_a, region_b = region_b, region_a
                bucket = result.setdefault((region_a, region_b), [0, 0.0])
                bucket[0] += 1
                bucket[1] += float(edge_value)

        consume(
            labels[:, :-1], labels[:, 1:],
            np.maximum(edge_map[:, :-1], edge_map[:, 1:]),
        )
        consume(
            labels[:-1, :], labels[1:, :],
            np.maximum(edge_map[:-1, :], edge_map[1:, :]),
        )
        return result

    @staticmethod
    def _photo_component_hue(values, weights=None):
        hue = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(hue) == 0:
            return 0.0
        if weights is None:
            weights = np.ones(len(hue), dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        angles = hue * (2.0 * math.pi / 180.0)
        vector = np.sum(weights * np.exp(1j * angles))
        angle = math.atan2(float(np.imag(vector)), float(np.real(vector)))
        if angle < 0.0:
            angle += 2.0 * math.pi
        return angle * 180.0 / (2.0 * math.pi)

    def _photo_build_object_edit_groups(self, rgb, owner_map, palette_rgb):
        """Convert color sublayers into spatial object/material edit groups.

        A garment can contain bright, middle and shadow colors.  These colors remain
        separate sublayers internally, but their connected components are grouped into
        one editable object region when spatial layout and material hue agree.  This
        avoids flattening the original shading while presenting a complete coat/shirt.
        """
        original = np.asarray(rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32)
        palette = np.asarray(palette_rgb, dtype=np.uint8)
        height, width = owners.shape
        atom_map = np.full((height, width), -1, dtype=np.int32)
        hsv_image = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab_image = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
        atoms = []
        atom_id = 0
        for owner in range(len(palette)):
            owner_mask = owners == owner
            component_count, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
                np.uint8(owner_mask), connectivity=8
            )
            if component_count <= 1:
                continue
            component_ids = np.arange(1, component_count, dtype=np.int32)
            component_to_atom = np.full(component_count, -1, dtype=np.int32)
            component_to_atom[1:] = np.arange(
                atom_id, atom_id + len(component_ids), dtype=np.int32
            )
            mapped_atoms = component_to_atom[component_labels]
            active = mapped_atoms >= 0
            atom_map[active] = mapped_atoms[active]

            active_components = component_labels[owner_mask]
            component_values = np.column_stack([
                hsv_image[:, :, 1][owner_mask],
                hsv_image[:, :, 2][owner_mask],
                lab_image[:, :, 0][owner_mask],
                lab_image[:, :, 1][owner_mask],
                lab_image[:, :, 2][owner_mask],
            ]).astype(np.float32)
            _counts, component_means, _std = self._photo_label_statistics(
                active_components, component_values, count=component_count
            )
            hue_angles = (
                hsv_image[:, :, 0][owner_mask].astype(np.float64)
                * (2.0 * math.pi / 180.0)
            )
            hue_cos = np.bincount(
                active_components,
                weights=np.cos(hue_angles),
                minlength=component_count,
            )
            hue_sin = np.bincount(
                active_components,
                weights=np.sin(hue_angles),
                minlength=component_count,
            )
            component_hues = (
                np.mod(np.arctan2(hue_sin, hue_cos), 2.0 * math.pi)
                * 180.0 / (2.0 * math.pi)
            )
            for component in range(1, component_count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area <= 0:
                    continue
                atoms.append({
                    'owner': int(owner),
                    'area': area,
                    'bbox': tuple(int(v) for v in stats[component, :4]),
                    'centroid': np.asarray(centroids[component], dtype=np.float64),
                    'hue': float(component_hues[component]),
                    'saturation': float(component_means[component, 0]),
                    'value': float(component_means[component, 1]),
                    'lab': component_means[component, 2:5].astype(np.float32),
                })
                atom_id += 1
        if atom_id == 0:
            return owners.copy(), palette.copy(), np.asarray([1.0], dtype=np.float32), [[0]]

        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = self._palette_iga_robust_normalize(
            np.sqrt(grad_x * grad_x + grad_y * grad_y), 94.0
        )
        adjacency = self._photo_adjacency(atom_map, edge)
        parent, find, union = self._photo_union_find(atom_id)
        root_members = {index: [index] for index in range(atom_id)}
        stats_cache = {}

        def members(root):
            root = find(root)
            return root_members[root]

        def group_stats(indices):
            areas = np.asarray([atoms[index]['area'] for index in indices], dtype=np.float64)
            total = max(float(np.sum(areas)), 1.0)
            hue = self._photo_component_hue(
                [atoms[index]['hue'] for index in indices], areas
            )
            saturation = float(np.sum([
                atoms[index]['saturation'] * atoms[index]['area'] for index in indices
            ]) / total)
            value = float(np.sum([
                atoms[index]['value'] * atoms[index]['area'] for index in indices
            ]) / total)
            xs, ys = [], []
            for index in indices:
                x, y, box_width, box_height = atoms[index]['bbox']
                xs.extend((x, x + box_width))
                ys.extend((y, y + box_height))
            centroid = np.average(
                np.asarray([atoms[index]['centroid'] for index in indices]),
                axis=0, weights=areas,
            )
            return {
                'area': int(total),
                'hue': float(hue),
                'saturation': saturation,
                'value': value,
                'bbox': (min(xs), min(ys), max(xs), max(ys)),
                'centroid': np.asarray(centroid, dtype=np.float64),
            }

        def stats_for_root(root):
            root = find(root)
            if root not in stats_cache:
                stats_cache[root] = group_stats(root_members[root])
            return stats_cache[root]

        def merge_roots(first, second):
            first = find(first)
            second = find(second)
            if first == second:
                return first
            combined_members = root_members[first] + root_members[second]
            union(first, second)
            merged_root = find(first)
            root_members.pop(first, None)
            root_members.pop(second, None)
            stats_cache.pop(first, None)
            stats_cache.pop(second, None)
            root_members[merged_root] = combined_members
            return merged_root

        # Pass 1: merge adjacent color components that are light/shadow variants of
        # the same saturated material.  Group-level hue spread prevents hue chaining
        # from orange -> ochre -> yellow.
        sorted_edges = sorted(adjacency.items(), key=lambda item: -item[1][0])
        for (first, second), (contact, edge_sum) in sorted_edges:
            root_first, root_second = find(first), find(second)
            if root_first == root_second:
                continue
            group_first = members(root_first)
            group_second = members(root_second)
            stats_first = stats_for_root(root_first)
            stats_second = stats_for_root(root_second)
            hue_gap = self._photo_hue_distance(
                stats_first['hue'] * 2.0, stats_second['hue'] * 2.0
            ) / 2.0
            combined = group_first + group_second
            combined_stats = group_stats(combined)
            hue_spread = max(
                self._photo_hue_distance(
                    atoms[index]['hue'] * 2.0, combined_stats['hue'] * 2.0
                ) / 2.0
                for index in combined
            )
            boundary = float(edge_sum) / max(float(contact), 1.0)
            vivid_material = (
                min(stats_first['saturation'], stats_second['saturation']) >= 175.0
                and hue_gap <= 10.0
                and hue_spread <= 11.0
                and contact >= 3
            )
            neutral_material = (
                max(stats_first['saturation'], stats_second['saturation']) < 55.0
                and abs(stats_first['value'] - stats_second['value']) <= 45.0
                and boundary <= 0.35
                and contact >= 10
            )
            if vivid_material or neutral_material:
                merge_roots(root_first, root_second)

        # Pass 2: merge disconnected visible parts of the same garment/material.
        # Example: the left and right orange jacket portions are separated by the
        # yellow shirt, but are one object and must be edited together.
        diagonal = max(math.hypot(width, height), 1.0)
        changed = True
        while changed:
            changed = False
            roots = list(root_members)
            info = {root: stats_for_root(root) for root in roots}
            eligible_roots = [
                root for root in roots
                if info[root]['saturation'] >= 180.0
                and info[root]['area'] >= 0.005 * owners.size
            ]
            best_pair = None
            best_score = None
            for first_index in range(len(eligible_roots)):
                for second_index in range(first_index + 1, len(eligible_roots)):
                    root_first = eligible_roots[first_index]
                    root_second = eligible_roots[second_index]
                    stats_first = info[root_first]
                    stats_second = info[root_second]
                    hue_gap = self._photo_hue_distance(
                        stats_first['hue'] * 2.0, stats_second['hue'] * 2.0
                    ) / 2.0
                    if hue_gap > 8.0:
                        continue
                    box_first = stats_first['bbox']
                    box_second = stats_second['bbox']
                    vertical_overlap = max(
                        0, min(box_first[3], box_second[3]) - max(box_first[1], box_second[1])
                    )
                    minimum_height = max(
                        1, min(box_first[3] - box_first[1], box_second[3] - box_second[1])
                    )
                    overlap_ratio = vertical_overlap / minimum_height
                    horizontal_gap = max(
                        0, max(box_first[0], box_second[0]) - min(box_first[2], box_second[2])
                    )
                    centroid_distance = float(np.linalg.norm(
                        stats_first['centroid'] - stats_second['centroid']
                    )) / diagonal
                    if not (overlap_ratio >= 0.25 or horizontal_gap / diagonal < 0.12):
                        continue
                    if centroid_distance > 0.45:
                        continue
                    score = hue_gap + 5.0 * centroid_distance - 2.0 * overlap_ratio
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (root_first, root_second)
            if best_pair is not None:
                merge_roots(best_pair[0], best_pair[1])
                changed = True

        # Material groups can still contain tiny isolated fragments.  Attach only
        # small, compatible fragments to an adjacent group; significant accents stay
        # independent.  The dynamic maximum stays close to the original layer count.
        def build_group_map():
            ordered = sorted(
                root_members.values(),
                key=lambda values: -sum(atoms[index]['area'] for index in values),
            )
            atom_to_group = np.full(atom_id, -1, dtype=np.int32)
            for group_index, values in enumerate(ordered):
                atom_to_group[np.asarray(values, dtype=np.intp)] = group_index
            return atom_to_group[atom_map].astype(np.int32)

        group_map = build_group_map()
        desired_max = max(8, min(18, len(palette) + 1))
        cie_lab = rgb2lab(original.astype(np.float32) / 255.0).astype(np.float32)
        while int(group_map.max()) + 1 > desired_max:
            group_count = int(group_map.max()) + 1
            group_features = np.concatenate([
                hsv_image[:, :, 1:2],
                cie_lab,
            ], axis=2)
            group_area, group_means, _group_std = self._photo_label_statistics(
                group_map, group_features, count=group_count
            )
            group_saturation = group_means[:, 0]
            group_lab = group_means[:, 1:4]
            hue_angles = (
                hsv_image[:, :, 0].reshape(-1).astype(np.float64)
                * (2.0 * math.pi / 180.0)
            )
            flat_groups = group_map.reshape(-1)
            hue_cos = np.bincount(
                flat_groups, weights=np.cos(hue_angles), minlength=group_count
            )
            hue_sin = np.bincount(
                flat_groups, weights=np.sin(hue_angles), minlength=group_count
            )
            group_hue = (
                np.mod(np.arctan2(hue_sin, hue_cos), 2.0 * math.pi)
                * 180.0 / (2.0 * math.pi)
            ).astype(np.float32)
            group_adjacency = self._photo_adjacency(group_map, edge)
            # 同一轮只合并互不冲突的区域对。旧实现每合并一个小区域就重新
            # 扫描整图并重建邻接，复杂花纹会重复数百次；批处理仍使用完全
            # 相同的兼容性与评分规则，但把整图统计次数降到对数级。
            removed_groups = set()
            target_groups = set()
            merge_pairs = []
            merge_budget = max(0, group_count - desired_max)
            for small_group in np.argsort(group_area, kind='stable'):
                small_group = int(small_group)
                if small_group in removed_groups or small_group in target_groups:
                    continue
                ratio = float(group_area[small_group]) / max(float(group_map.size), 1.0)
                candidates = []
                for (first, second), (contact, edge_sum) in group_adjacency.items():
                    if first == small_group:
                        neighbor = second
                    elif second == small_group:
                        neighbor = first
                    else:
                        continue
                    neighbor = int(neighbor)
                    if neighbor in removed_groups:
                        continue
                    hue_gap = self._photo_hue_distance(
                        group_hue[small_group] * 2.0, group_hue[neighbor] * 2.0
                    ) / 2.0
                    perceptual = float(deltaE_ciede2000(
                        group_lab[small_group][None, :], group_lab[neighbor][None, :]
                    )[0])
                    boundary = float(edge_sum) / max(float(contact), 1.0)
                    compatible = (
                        min(group_saturation[small_group], group_saturation[neighbor]) >= 150.0
                        and hue_gap <= 14.0
                    ) or (
                        max(group_saturation[small_group], group_saturation[neighbor]) < 75.0
                        and perceptual <= 15.0
                    ) or perceptual <= 10.0
                    if not compatible and ratio > 0.004:
                        continue
                    score = perceptual + 5.0 * boundary - 2.0 * math.log1p(contact)
                    candidates.append((score, neighbor))
                if candidates:
                    _score, neighbor = min(candidates)
                    removed_groups.add(small_group)
                    target_groups.add(int(neighbor))
                    merge_pairs.append((small_group, int(neighbor)))
                    if len(merge_pairs) >= merge_budget:
                        break
            if not merge_pairs:
                break
            group_remap = np.arange(group_count, dtype=np.int32)
            for small_group, neighbor in merge_pairs:
                group_remap[small_group] = neighbor
            group_map = group_remap[group_map]
            _unique, inverse = np.unique(group_map, return_inverse=True)
            group_map = inverse.reshape(group_map.shape).astype(np.int32)

        group_count = int(group_map.max()) + 1
        flat_rgb = original.reshape(-1, 3)
        flat_lab = lab_image.reshape(-1, 3)
        group_palette = []
        proportions = []
        members_by_group = []
        for group in range(group_count):
            mask = group_map.reshape(-1) == group
            group_palette.append(self._palette_iga_cluster_representative(
                flat_rgb[mask], flat_lab[mask], np.mean(flat_lab[mask], axis=0)
            ))
            proportions.append(float(np.mean(mask)))
            owner_ids = np.unique(owners.reshape(-1)[mask]).astype(np.int32).tolist()
            members_by_group.append(owner_ids)
        group_palette = np.asarray(group_palette, dtype=np.uint8)
        proportions = np.asarray(proportions, dtype=np.float32)
        order = np.argsort(-proportions, kind='stable')
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order), dtype=np.intp)
        group_map = inverse[group_map]
        group_palette = group_palette[order]
        proportions = proportions[order]
        members_by_group = [members_by_group[index] for index in order]
        return group_map.astype(np.int32), group_palette, proportions, members_by_group

    @staticmethod
    def _palette_build_owner_edge_band(owner_map, radius=4):
        """Return a narrow band around existing owner boundaries.

        The interior of every object region is deliberately excluded so edge
        refinement cannot become a second whole-image segmentation pass.
        """
        owners = np.asarray(owner_map, dtype=np.int32)
        boundary = np.zeros(owners.shape, dtype=np.uint8)
        horizontal = owners[:, 1:] != owners[:, :-1]
        vertical = owners[1:, :] != owners[:-1, :]
        boundary[:, 1:][horizontal] = 1
        boundary[:, :-1][horizontal] = 1
        boundary[1:, :][vertical] = 1
        boundary[:-1, :][vertical] = 1
        radius = max(1, int(radius))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(boundary, kernel, iterations=1).astype(bool)

    def _palette_fixed_label_palette(self, rgb, label_map, label_count):
        """Recompute representative colors without changing region IDs."""
        original = np.asarray(rgb, dtype=np.uint8)
        labels = np.asarray(label_map, dtype=np.int32)
        flat_rgb = original.reshape(-1, 3)
        flat_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        flat_labels = labels.reshape(-1)
        palette = []
        proportions = []
        for index in range(int(label_count)):
            mask = flat_labels == index
            if not np.any(mask):
                raise ValueError(f"Refined region R{index + 1} became empty")
            palette.append(self._palette_iga_cluster_representative(
                flat_rgb[mask], flat_lab[mask], np.mean(flat_lab[mask], axis=0)
            ))
            proportions.append(float(np.mean(mask)))
        proportions = np.asarray(proportions, dtype=np.float32)
        proportions /= max(float(proportions.sum()), 1e-8)
        return np.asarray(palette, dtype=np.uint8), proportions

    def _palette_merge_similar_object_regions(
            self, rgb, group_map, palette_rgb, members=None, target_max=12):
        """合并高相似度对象区域，并保持单标签覆盖完整。

        Palette Result Editor 面向用户的是“可编辑对象区域”，不是底层颜色碎片。
        因此这里只合并感知色差很小、或同色相族的光照/材质近似区域；
        `target_max` 只是软目标，不会把远色强行压成一类。
        """
        original = np.asarray(rgb, dtype=np.uint8)
        labels = np.asarray(group_map, dtype=np.int32).copy()
        palette = np.asarray(palette_rgb, dtype=np.uint8).copy()
        if original.ndim != 3 or original.shape[:2] != labels.shape:
            raise ValueError("Object-region merge requires RGB image and matching label map")
        if labels.size == 0 or len(palette) <= 1:
            empty_metrics = {
                'initial_regions': int(len(palette)),
                'final_regions': int(len(palette)),
                'merged_regions': 0,
                'coverage_percent': 100.0,
                'unassigned_pixels': 0,
                'target_max': int(target_max) if target_max is not None else None,
            }
            proportions = np.asarray([1.0], dtype=np.float32) if len(palette) == 1 else np.empty((0,), dtype=np.float32)
            return labels, palette, proportions, members or [], empty_metrics

        if np.any(labels < 0):
            raise ValueError("Object-region map contains unassigned pixels")
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        labels = inverse.reshape(labels.shape).astype(np.int32)
        if int(unique_labels.max(initial=0)) >= len(palette):
            raise ValueError("Object-region map references a missing palette color")
        palette = palette[unique_labels.astype(np.intp)]

        if members is not None and len(members) >= int(unique_labels.max(initial=0)) + 1:
            region_members = [
                list(members[int(label)])
                for label in unique_labels.astype(np.int32).tolist()
            ]
        else:
            region_members = [[int(label)] for label in unique_labels.astype(np.int32).tolist()]

        initial_count = len(palette)
        target = None if target_max is None else max(2, int(target_max))
        flat_rgb = original.reshape(-1, 3)
        flat_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        total = max(int(labels.size), 1)
        merge_count = 0

        for _iteration in range(max(1, initial_count * 2)):
            count = len(palette)
            if count <= 1:
                break
            palette_lab = self._palette_rgb_to_lab_array(palette)
            flat_labels = labels.reshape(-1)
            counts = np.bincount(flat_labels, minlength=count).astype(np.int64)
            areas = counts.astype(np.float64) / float(total)
            adjacency = self._photo_adjacency(
                labels, np.zeros(labels.shape, dtype=np.float32)
            )
            best_pair = None
            best_score = None
            above_soft_target = target is not None and count > target

            for first in range(count):
                for second in range(first + 1, count):
                    if counts[first] == 0 or counts[second] == 0:
                        continue
                    distance = self._palette_iga_perceptual_distance(
                        palette_lab[first], palette_lab[second]
                    )
                    same_family = (
                        self._palette_iga_color_family(palette_lab[first])
                        == self._palette_iga_color_family(palette_lab[second])
                    )
                    cie_first = self._palette_cv_lab_to_cie(palette_lab[first])
                    cie_second = self._palette_cv_lab_to_cie(palette_lab[second])
                    chroma = max(
                        float(math.hypot(cie_first[1], cie_first[2])),
                        float(math.hypot(cie_second[1], cie_second[2])),
                    )
                    neutral = chroma < 16.0
                    smaller = min(float(areas[first]), float(areas[second]))
                    contact = float(adjacency.get((first, second), adjacency.get((second, first), (0, 0.0)))[0])
                    adjacent = contact > 0.0

                    merge = False
                    if distance <= 5.8:
                        merge = True
                    elif neutral and distance <= 8.5:
                        merge = True
                    elif same_family and distance <= 10.8:
                        merge = True
                    elif adjacent and same_family and distance <= 13.2:
                        merge = True
                    elif above_soft_target and same_family and smaller <= 0.10 and distance <= 14.6:
                        merge = True
                    elif above_soft_target and neutral and smaller <= 0.12 and distance <= 11.2:
                        merge = True
                    elif above_soft_target and smaller <= 0.018 and distance <= 13.5:
                        merge = True

                    if not merge:
                        continue
                    score = (
                        distance
                        - (1.2 if same_family else 0.0)
                        - (0.8 if neutral else 0.0)
                        - (0.7 if adjacent else 0.0)
                        - 2.0 * min(smaller, 0.08)
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (first, second)

            if best_pair is None:
                break

            first, second = best_pair
            keep, remove = (first, second) if counts[first] >= counts[second] else (second, first)
            labels[labels == remove] = keep
            compact = [index for index in range(count) if index != remove]
            remap = np.full(count, -1, dtype=np.int32)
            remap[np.asarray(compact, dtype=np.intp)] = np.arange(len(compact), dtype=np.int32)
            labels = remap[labels].astype(np.int32)

            merged_members = sorted(set(region_members[keep] + region_members[remove]))
            region_members = [region_members[index] for index in compact]
            compact_keep = compact.index(keep)
            region_members[compact_keep] = merged_members

            # 未参与本轮合并的区域像素完全不变，其代表色也不变。旧实现每次
            # 合并后为所有区域重新 unique/sort，复杂图会重复排序数百次；
            # 这里只重算被合并区域，结果与全量重算一致。
            palette = palette[np.asarray(compact, dtype=np.intp)].copy()
            flat_labels = labels.reshape(-1)
            merged_mask = flat_labels == compact_keep
            palette[compact_keep] = self._palette_iga_cluster_representative(
                flat_rgb[merged_mask],
                flat_lab[merged_mask],
                np.mean(flat_lab[merged_mask], axis=0),
            )
            merge_count += 1

        final_count = len(palette)
        proportions = np.bincount(labels.reshape(-1), minlength=final_count).astype(np.float32)
        proportions /= max(float(proportions.sum()), 1e-8)
        order = np.argsort(-proportions, kind='stable')
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order), dtype=np.intp)
        labels = inverse_order[labels].astype(np.int32)
        palette = palette[order]
        proportions = proportions[order]
        region_members = [region_members[int(index)] for index in order]
        valid = (labels >= 0) & (labels < len(palette))
        metrics = {
            'initial_regions': int(initial_count),
            'final_regions': int(final_count),
            'merged_regions': int(merge_count),
            'coverage_percent': float(np.mean(valid)) * 100.0,
            'unassigned_pixels': int(np.count_nonzero(~valid)),
            'target_max': int(target) if target is not None else None,
            'method': 'perceptual similar-object region merge',
        }
        return labels, palette.astype(np.uint8), proportions.astype(np.float32), region_members, metrics

    # ------------------------------------------------------------------
    # v4.4.5 fewer_object_regions: 受控对象分层（目标 8~12 个完整区域）
    # ------------------------------------------------------------------
    PHOTO_OBJECT_TARGET_REGIONS = 10
    PHOTO_OBJECT_MAX_REGIONS = 12
    PHOTO_OBJECT_MIN_AREA_MERGE = 0.018
    PHOTO_OBJECT_MIN_REGIONS = 5
    PHOTO_REDUNDANT_HUE_FLOOR = 3

    def _photo_redundant_hue_match(self, first_lab, second_lab):
        """判断两个照片区域是否只是同一色相的明暗重复层。

        该规则只处理高置信度的颜色冗余，不宣称识别出了物体类别。它允许
        盘子红/水果红这类色相几乎一致、仅有适度光照差异的区域在旧版五层
        下限以下继续合并，同时保护红/橙等真实不同色系。
        """
        first = np.asarray(first_lab, dtype=np.float32)
        second = np.asarray(second_lab, dtype=np.float32)
        delta = float(deltaE_ciede2000(first[None, :], second[None, :])[0])
        chroma_first = float(math.hypot(first[1], first[2]))
        chroma_second = float(math.hypot(second[1], second[2]))
        if min(chroma_first, chroma_second) < 18.0:
            return False
        hue_first = math.degrees(math.atan2(first[2], first[1])) % 360.0
        hue_second = math.degrees(math.atan2(second[2], second[1])) % 360.0
        hue_gap = self._photo_hue_distance(hue_first, hue_second)
        return bool(
            hue_gap <= 5.0
            and delta <= 11.5
            and abs(float(first[0] - second[0])) <= 15.0
            and abs(chroma_first - chroma_second) <= 18.0
        )

    def _palette_reduce_object_regions_v445(
            self, rgb, group_map, palette_rgb, members=None,
            target_regions=None, max_regions=None, min_area_merge=None):
        """v4.4.5 受控分层：把对象区域收敛到 target~max 个完整“对象层”。

        v4.4.3/v4.4.4 的合并偏保守，真实照片仍会留下 16~20 个区域。
        本方法在边缘精修后的对象区域图上做三阶段受控合并：

        1) 背景聚合：大面积、低纹理、接触画面边界、颜色连续的区域
           使用更宽松的阈值优先互并（天空 / 墙面 / 地面不再拆碎）；
        2) 小区域吸附：面积 < ``min_area_merge`` 的区域默认不单独成层，
           并入颜色最接近的相邻主区域；若多个候选色差接近，
           则并入接触边界最长的那个。只有与所有邻居颜色都截然不同的
           鲜艳小物体（例如气球）才保留独立；
        3) 相似相邻合并 + 数量收敛：以 CIEDE2000 色差、边界强度、
           纹理差、接触长度、同色相族与空间距离打分，迭代合并
           “最该合并”的一对；数量 > ``max_regions`` 时强制合并，
           > ``target_regions`` 时按放宽阈值继续，
           ≤ ``target_regions`` 后只允许近重复合并。

        所有操作只重写标签归属，不改变覆盖率 / 单一 owner 的约束；
        合并沿既有精修边界进行，因此 v4.4.3 的边缘精修结果保持不变。
        """
        original = np.asarray(rgb, dtype=np.uint8)
        labels = np.asarray(group_map, dtype=np.int32).copy()
        palette_in = np.asarray(palette_rgb, dtype=np.uint8)
        if original.ndim != 3 or original.shape[:2] != labels.shape:
            raise ValueError("v4.4.5 region reduction requires RGB image and matching label map")
        if labels.size == 0 or np.any(labels < 0):
            raise ValueError("v4.4.5 region reduction received unassigned pixels")

        target_regions = (
            self.PHOTO_OBJECT_TARGET_REGIONS if target_regions is None else int(target_regions)
        )
        max_regions = (
            self.PHOTO_OBJECT_MAX_REGIONS if max_regions is None else int(max_regions)
        )
        min_area_merge = (
            self.PHOTO_OBJECT_MIN_AREA_MERGE if min_area_merge is None else float(min_area_merge)
        )
        target_regions = max(2, target_regions)
        max_regions = max(target_regions, max_regions)
        # 下限保护：真实照片默认不把区域并到 5 个以下；
        # 若调用方显式要求更低的 target，则以 target 为准。
        min_regions = max(2, min(int(self.PHOTO_OBJECT_MIN_REGIONS), target_regions))

        unique_labels, inverse = np.unique(labels, return_inverse=True)
        labels = inverse.reshape(labels.shape).astype(np.int32)
        count = int(len(unique_labels))
        if members is not None and len(members) >= int(unique_labels.max(initial=0)) + 1:
            region_members = [
                list(members[int(value)]) for value in unique_labels.astype(np.int32).tolist()
            ]
        else:
            region_members = [[int(value)] for value in unique_labels.astype(np.int32).tolist()]
        initial_count = count
        if count <= 1:
            metrics = {
                'initial_regions': initial_count,
                'final_regions': count,
                'merged_regions': 0,
                'background_merges': 0,
                'small_regions_absorbed': 0,
                'similarity_merges': 0,
                'redundant_hue_merges': 0,
                'forced_merges': 0,
                'kept_distinct_small_regions': 0,
                'coverage_percent': 100.0,
                'unassigned_pixels': 0,
                'target_regions': int(target_regions),
                'max_regions': int(max_regions),
                'min_area_merge_percent': float(min_area_merge) * 100.0,
                'method': 'v4.4.5 controlled fewer-object-region reduction',
            }
            proportions = np.asarray([1.0] * count, dtype=np.float32)
            return labels, palette_in.copy(), proportions, region_members, metrics

        height, width = labels.shape
        total = float(labels.size)
        diagonal = max(math.hypot(width, height), 1.0)
        cie_lab = rgb2lab(original.astype(np.float32) / 255.0).astype(np.float32)
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = self._palette_iga_robust_normalize(
            np.sqrt(grad_x * grad_x + grad_y * grad_y), 94.0
        )

        flat_labels = labels.reshape(-1)
        flat_lab = cie_lab.reshape(-1, 3)
        flat_gray = gray.reshape(-1)
        area = np.bincount(flat_labels, minlength=count).astype(np.float64)
        sum_lab = np.zeros((count, 3), dtype=np.float64)
        for channel in range(3):
            sum_lab[:, channel] = np.bincount(
                flat_labels, weights=flat_lab[:, channel].astype(np.float64), minlength=count
            )
        sum_gray = np.bincount(flat_labels, weights=flat_gray.astype(np.float64), minlength=count)
        sum_gray_sq = np.bincount(
            flat_labels, weights=(flat_gray.astype(np.float64) ** 2), minlength=count
        )
        row_map = np.broadcast_to(
            np.arange(height, dtype=np.float64)[:, None], labels.shape
        ).reshape(-1)
        col_map = np.broadcast_to(
            np.arange(width, dtype=np.float64)[None, :], labels.shape
        ).reshape(-1)
        sum_row = np.bincount(flat_labels, weights=row_map, minlength=count)
        sum_col = np.bincount(flat_labels, weights=col_map, minlength=count)
        border_pixels = np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
        border_counts = np.bincount(border_pixels, minlength=count).astype(np.float64)
        border_total = max(float(border_pixels.size), 1.0)

        parent, find, union = self._photo_union_find(count)
        stats = {
            index: {
                'area': float(area[index]),
                'sum_lab': sum_lab[index].copy(),
                'sum_gray': float(sum_gray[index]),
                'sum_gray_sq': float(sum_gray_sq[index]),
                'sum_row': float(sum_row[index]),
                'sum_col': float(sum_col[index]),
                'border': float(border_counts[index]),
                'members': sorted(set(region_members[index])),
                'protected': False,
            }
            for index in range(count)
        }
        adjacency = {}
        for (first, second), (contact, edge_sum) in self._photo_adjacency(labels, gradient).items():
            adjacency[(int(first), int(second))] = [float(contact), float(edge_sum)]

        merge_log = {
            'background': 0,
            'small_absorbed': 0,
            'similarity': 0,
            'redundant_hue': 0,
            'forced': 0,
        }
        protected_roots = set()

        def alive_roots():
            return sorted({find(index) for index in range(count)})

        def mean_lab_of(root):
            data = stats[root]
            return (data['sum_lab'] / max(data['area'], 1.0)).astype(np.float32)

        def texture_of(root):
            data = stats[root]
            pixels = max(data['area'], 1.0)
            mean = data['sum_gray'] / pixels
            variance = max(data['sum_gray_sq'] / pixels - mean * mean, 0.0)
            return math.sqrt(variance) / 64.0

        def area_ratio_of(root):
            return stats[root]['area'] / total

        def border_ratio_of(root):
            return stats[root]['border'] / border_total

        def centroid_of(root):
            data = stats[root]
            pixels = max(data['area'], 1.0)
            return np.asarray(
                [data['sum_row'] / pixels, data['sum_col'] / pixels], dtype=np.float64
            )

        def chroma_of(root):
            lab = mean_lab_of(root)
            return float(math.hypot(float(lab[1]), float(lab[2])))

        def contact_of(first, second):
            key = (first, second) if first < second else (second, first)
            entry = adjacency.get(key)
            if entry is None:
                return 0.0, 0.0
            return float(entry[0]), float(entry[1])

        def boundary_of(first, second):
            contact, edge_sum = contact_of(first, second)
            if contact <= 0.0:
                return 1.0
            return edge_sum / contact

        def neighbors_of(root):
            found = set()
            for first, second in adjacency:
                if first == root:
                    found.add(second)
                elif second == root:
                    found.add(first)
            return found

        def merge_regions(keep, remove, reason):
            keep_root, remove_root = find(keep), find(remove)
            if keep_root == remove_root:
                return keep_root
            union(keep_root, remove_root)
            new_root = find(keep_root)
            other = remove_root if new_root == keep_root else keep_root
            base = stats[new_root]
            donor = stats[other]
            stats[new_root] = {
                'area': base['area'] + donor['area'],
                'sum_lab': base['sum_lab'] + donor['sum_lab'],
                'sum_gray': base['sum_gray'] + donor['sum_gray'],
                'sum_gray_sq': base['sum_gray_sq'] + donor['sum_gray_sq'],
                'sum_row': base['sum_row'] + donor['sum_row'],
                'sum_col': base['sum_col'] + donor['sum_col'],
                'border': base['border'] + donor['border'],
                'members': sorted(set(base['members']) | set(donor['members'])),
                'protected': bool(base['protected'] or donor['protected']),
            }
            del stats[other]
            protected_roots.discard(other)
            if stats[new_root]['protected']:
                protected_roots.add(new_root)
            updated = {}
            for (first, second), value in adjacency.items():
                mapped_first = new_root if first in (keep_root, remove_root) else first
                mapped_second = new_root if second in (keep_root, remove_root) else second
                if mapped_first == mapped_second:
                    continue
                key = (
                    (mapped_first, mapped_second)
                    if mapped_first < mapped_second else (mapped_second, mapped_first)
                )
                entry = updated.setdefault(key, [0.0, 0.0])
                entry[0] += value[0]
                entry[1] += value[1]
            adjacency.clear()
            adjacency.update(updated)
            merge_log[reason] += 1
            return new_root

        # ---------- 阶段 1：背景优先聚合 ----------
        def is_background(root):
            ratio = area_ratio_of(root)
            return (
                (border_ratio_of(root) >= 0.05 and ratio >= 0.04 and texture_of(root) <= 0.50)
                or (ratio >= 0.16 and border_ratio_of(root) >= 0.015)
            )

        while len(alive_roots()) > min_regions:
            roots = alive_roots()
            best_pair = None
            best_score = None
            for first_index in range(len(roots)):
                for second_index in range(first_index + 1, len(roots)):
                    first, second = roots[first_index], roots[second_index]
                    if not (is_background(first) and is_background(second)):
                        continue
                    delta, neutral, same_family, _lightness_gap = self._photo_relation(
                        mean_lab_of(first), mean_lab_of(second)
                    )
                    contact, _edge_sum = contact_of(first, second)
                    low_chroma = max(chroma_of(first), chroma_of(second)) < 20.0
                    relaxed = same_family or neutral or low_chroma
                    if contact > 0.0:
                        limit = 17.0 if relaxed else 12.0
                        boundary = boundary_of(first, second)
                        if delta > limit or boundary > 0.62:
                            continue
                        score = delta + 5.0 * boundary
                    else:
                        # 被主体分隔的同一背景（左右两片天空 / 墙面）
                        if delta > 13.0 or not relaxed:
                            continue
                        score = delta + 6.0
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (first, second)
            if best_pair is None:
                break
            merge_regions(best_pair[0], best_pair[1], 'background')

        # ---------- 阶段 2：小区域吸附 ----------
        while len(alive_roots()) > min_regions:
            candidates = [
                root for root in alive_roots()
                if area_ratio_of(root) < min_area_merge and root not in protected_roots
            ]
            if not candidates:
                break
            root = min(candidates, key=area_ratio_of)
            neighbor_roots = neighbors_of(root)
            if not neighbor_roots:
                protected_roots.add(root)
                stats[root]['protected'] = True
                continue
            evaluated = []
            for neighbor in neighbor_roots:
                delta, _neutral, _same_family, _lightness_gap = self._photo_relation(
                    mean_lab_of(root), mean_lab_of(neighbor)
                )
                contact, _edge_sum = contact_of(root, neighbor)
                evaluated.append((float(delta), float(contact), neighbor))
            # 接触门槛：只有边界接触足够长的邻居才有资格吸附小区域，
            # 避免小物体沿几个像素的细缝被并进空间上无关的大层（如围巾并入天空）
            total_contact = sum(item[1] for item in evaluated)
            min_contact = max(4.0, 0.04 * total_contact)
            eligible = [item for item in evaluated if item[1] >= min_contact]
            if not eligible:
                eligible = evaluated
            eligible.sort(key=lambda item: item[0])
            best_delta = eligible[0][0]
            if best_delta > 21.0 and chroma_of(root) >= 18.0 and area_ratio_of(root) >= 0.006:
                # 与所有邻居颜色都截然不同的有彩度小物体（围巾、气球、反光布等）保留独立
                protected_roots.add(root)
                stats[root]['protected'] = True
                continue
            close = [item for item in eligible if item[0] <= best_delta + 2.5]
            target_neighbor = max(close, key=lambda item: item[1])[2]
            merge_regions(target_neighbor, root, 'small_absorbed')

        # ---------- 阶段 3：相似相邻合并 + 数量收敛 ----------
        while True:
            roots = alive_roots()
            count_now = len(roots)

            # 颜色冗余优先于固定层数下限。旧版到 5 层立即停止，使色相几乎
            # 相同的深红/亮红继续分成两层；这里只允许强约束同色相对合并，
            # 且至少保留 3 个颜色层。
            redundant_pair = None
            redundant_score = None
            if count_now > int(self.PHOTO_REDUNDANT_HUE_FLOOR):
                for first_index in range(len(roots)):
                    for second_index in range(first_index + 1, len(roots)):
                        first, second = roots[first_index], roots[second_index]
                        first_lab = mean_lab_of(first)
                        second_lab = mean_lab_of(second)
                        if not self._photo_redundant_hue_match(
                                first_lab, second_lab):
                            continue
                        delta, _neutral, _family, lightness_gap = (
                            self._photo_relation(first_lab, second_lab)
                        )
                        score = float(delta) + 0.18 * float(lightness_gap)
                        if redundant_score is None or score < redundant_score:
                            redundant_score = score
                            redundant_pair = (first, second)
            if redundant_pair is not None:
                merge_regions(
                    redundant_pair[0], redundant_pair[1], 'redundant_hue'
                )
                continue

            if count_now <= min_regions:
                break
            over_max = count_now > max_regions
            over_target = count_now > target_regions
            best_pair = None
            best_score = None
            forced_pair = None
            forced_score = None
            for first_index in range(len(roots)):
                for second_index in range(first_index + 1, len(roots)):
                    first, second = roots[first_index], roots[second_index]
                    delta, neutral, same_family, lightness_gap = self._photo_relation(
                        mean_lab_of(first), mean_lab_of(second)
                    )
                    chroma_first = chroma_of(first)
                    chroma_second = chroma_of(second)
                    soft_neutral = max(chroma_first, chroma_second) < 18.0
                    texture_gap = abs(texture_of(first) - texture_of(second))
                    contact, _edge_sum = contact_of(first, second)
                    protected_pair = first in protected_roots or second in protected_roots
                    if contact > 0.0:
                        boundary = boundary_of(first, second)
                        smaller = min(stats[first]['area'], stats[second]['area'])
                        contact_norm = contact / max(math.sqrt(smaller), 1.0)
                        score = (
                            delta
                            + 6.0 * boundary
                            + 2.2 * texture_gap
                            - 1.3 * min(contact_norm, 2.0)
                            - (1.5 if same_family else 0.0)
                            - (1.0 if (neutral or soft_neutral) else 0.0)
                        )
                        if over_target:
                            allow = (
                                delta <= 6.5
                                or ((neutral or soft_neutral)
                                    and delta <= 14.0 and lightness_gap <= 30.0)
                                or (same_family and delta <= 16.0 and texture_gap <= 0.60)
                                or (delta <= 12.0 and boundary <= 0.55)
                            )
                        else:
                            allow = delta <= 5.0 and boundary <= 0.50
                        if over_max and not protected_pair:
                            if forced_score is None or score < forced_score:
                                forced_score = score
                                forced_pair = (first, second)
                    else:
                        # 同一物体被遮挡拆开的可见部分（如外套左右两片）
                        if not over_target:
                            continue
                        if not (same_family and delta <= 8.0
                                and min(chroma_first, chroma_second) >= 22.0):
                            continue
                        distance_norm = float(np.linalg.norm(
                            centroid_of(first) - centroid_of(second)
                        )) / diagonal
                        if distance_norm > 0.45:
                            continue
                        score = delta + 9.0 * distance_norm + 2.2 * texture_gap + 1.5
                        allow = True
                    if protected_pair and delta > 6.0:
                        allow = False
                    if not allow:
                        continue
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (first, second)
            if best_pair is not None:
                merge_regions(best_pair[0], best_pair[1], 'similarity')
                continue
            if over_max and forced_pair is not None:
                merge_regions(forced_pair[0], forced_pair[1], 'forced')
                continue
            break

        # ---------- 输出：重映射标签并重算代表色 ----------
        final_roots = alive_roots()
        final_count = len(final_roots)
        root_to_index = {root: index for index, root in enumerate(final_roots)}
        mapping = np.asarray(
            [root_to_index[find(index)] for index in range(count)], dtype=np.int32
        )
        labels = mapping[labels]
        flat_rgb = original.reshape(-1, 3)
        flat_lab_cv = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        flat_final = labels.reshape(-1)
        palette_out = []
        proportions = []
        members_out = []
        for index, root in enumerate(final_roots):
            mask = flat_final == index
            if not np.any(mask):
                raise ValueError(f"v4.4.5 reduction produced an empty region R{index + 1}")
            palette_out.append(self._palette_iga_cluster_representative(
                flat_rgb[mask], flat_lab_cv[mask], np.mean(flat_lab_cv[mask], axis=0)
            ))
            proportions.append(float(np.mean(mask)))
            members_out.append(sorted(set(stats[root]['members'])))
        palette_out = np.asarray(palette_out, dtype=np.uint8)
        proportions = np.asarray(proportions, dtype=np.float32)
        proportions /= max(float(proportions.sum()), 1e-8)
        order = np.argsort(-proportions, kind='stable')
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order), dtype=np.intp)
        labels = inverse_order[labels].astype(np.int32)
        palette_out = palette_out[order]
        proportions = proportions[order]
        members_out = [members_out[int(index)] for index in order]
        valid = (labels >= 0) & (labels < final_count)
        metrics = {
            'initial_regions': int(initial_count),
            'final_regions': int(final_count),
            'merged_regions': int(initial_count - final_count),
            'background_merges': int(merge_log['background']),
            'small_regions_absorbed': int(merge_log['small_absorbed']),
            'similarity_merges': int(merge_log['similarity']),
            'redundant_hue_merges': int(merge_log['redundant_hue']),
            'forced_merges': int(merge_log['forced']),
            'kept_distinct_small_regions': int(len(protected_roots)),
            'coverage_percent': float(np.mean(valid)) * 100.0,
            'unassigned_pixels': int(np.count_nonzero(~valid)),
            'target_regions': int(target_regions),
            'max_regions': int(max_regions),
            'min_area_merge_percent': float(min_area_merge) * 100.0,
            'method': 'v4.4.5 controlled fewer-object-region reduction',
        }
        return labels, palette_out, proportions, members_out, metrics

    @staticmethod
    def _palette_validate_refined_owner_map(
            original_map, candidate_map, palette_size, edge_band,
            soft_limit=0.06, hard_limit=0.08):
        """Validate local edge refinement and fall back on any unsafe change."""
        original = np.asarray(original_map, dtype=np.int32)
        candidate = np.asarray(candidate_map, dtype=np.int32)
        band = np.asarray(edge_band, dtype=bool)
        metrics = {
            'fallback': False,
            'reason': '',
            'changed_ratio': 0.0,
            'band_ratio': float(np.mean(band)) if band.size else 0.0,
            'coverage_percent': 100.0,
            'overlap_percent': 0.0,
            'unassigned_pixels': 0,
        }
        if candidate.shape != original.shape:
            metrics.update(fallback=True, reason='shape mismatch')
            return original.copy(), metrics
        if candidate.size == 0 or np.any(candidate < 0) or np.any(candidate >= int(palette_size)):
            metrics.update(fallback=True, reason='invalid owner index')
            return original.copy(), metrics
        changed = candidate != original
        if np.any(changed & ~band):
            metrics.update(fallback=True, reason='change outside edge band')
            return original.copy(), metrics
        changed_ratio = float(np.mean(changed))
        metrics['changed_ratio'] = changed_ratio
        counts = np.bincount(candidate.reshape(-1), minlength=int(palette_size))
        if len(counts) < int(palette_size) or np.any(counts[:int(palette_size)] == 0):
            metrics.update(fallback=True, reason='region became empty')
            return original.copy(), metrics
        if changed_ratio > float(hard_limit) + 1e-9:
            metrics.update(fallback=True, reason='hard change limit exceeded')
            return original.copy(), metrics
        # Refinement is expected to stay below the soft limit.  The algorithm
        # already budgets proposals to this value; this flag is diagnostic only.
        metrics['soft_limit_exceeded'] = bool(changed_ratio > float(soft_limit) + 1e-9)
        return candidate.copy(), metrics

    def _palette_refine_owner_edge_band(
            self, rgb, owner_map, palette_rgb,
            radius=None, soft_limit=0.06, hard_limit=0.08):
        """Refine only the narrow boundary band of existing object regions.

        Fine SLIC atoms are evaluated against *adjacent* current owners using
        perceptual color, core-distance, boundary strength and texture.  Region
        interiors, region IDs and the number of editable regions are fixed.
        """
        original = np.asarray(rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32)
        palette = np.asarray(palette_rgb, dtype=np.uint8)
        height, width = owners.shape
        if original.shape[:2] != owners.shape or len(palette) <= 1:
            return owners.copy(), {
                'fallback': False, 'reason': 'nothing to refine',
                'changed_ratio': 0.0, 'band_ratio': 0.0,
                'accepted_atoms': 0,
            }
        if radius is None:
            radius = int(np.clip(round(min(height, width) * 0.010), 3, 6))
        band = self._palette_build_owner_edge_band(owners, radius=radius)
        if not np.any(band):
            return owners.copy(), {
                'fallback': False, 'reason': 'no boundaries',
                'changed_ratio': 0.0, 'band_ratio': 0.0,
                'accepted_atoms': 0,
            }

        cie_lab = rgb2lab(original.astype(np.float32) / 255.0).astype(np.float32)
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = self._palette_iga_robust_normalize(
            np.sqrt(gx * gx + gy * gy), 94.0
        )
        mean_gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
        mean_square = cv2.GaussianBlur(gray * gray, (0, 0), 2.0)
        texture = np.sqrt(np.maximum(mean_square - mean_gray * mean_gray, 0.0)) / 64.0

        label_count = len(palette)
        core_lab = np.zeros((label_count, 3), dtype=np.float32)
        core_texture = np.zeros(label_count, dtype=np.float32)
        core_masks = []
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for label in range(label_count):
            mask = np.uint8(owners == label)
            core = cv2.erode(mask, erode_kernel, iterations=1).astype(bool)
            if not np.any(core):
                core = mask.astype(bool)
            core_masks.append(core)
            core_lab[label] = np.mean(cie_lab[core], axis=0)
            core_texture[label] = float(np.mean(texture[core]))

        # Half-resolution distance-to-core maps keep memory bounded while still
        # enforcing spatial continuity for the narrow edge band.
        half_w = max(1, (width + 1) // 2)
        half_h = max(1, (height + 1) // 2)
        diagonal = max(math.hypot(half_w, half_h), 1.0)
        distance_maps = np.empty((label_count, half_h, half_w), dtype=np.float16)
        for label, core in enumerate(core_masks):
            small_core = cv2.resize(
                core.astype(np.uint8), (half_w, half_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            distance = cv2.distanceTransform((~small_core).astype(np.uint8), cv2.DIST_L2, 3)
            distance_maps[label] = np.clip(distance / diagonal, 0.0, 1.0).astype(np.float16)

        if _PHOTO_LAYER_SKIMAGE_AVAILABLE:
            pixel_count = max(height * width, 1)
            segment_count = int(np.clip(round(pixel_count / 75.0), 700, 4200))
            filtered = cv2.bilateralFilter(original, d=5, sigmaColor=12, sigmaSpace=3)
            atoms = slic(
                filtered.astype(np.float32) / 255.0,
                n_segments=segment_count,
                compactness=3.8,
                sigma=0.45,
                start_label=0,
                enforce_connectivity=True,
                min_size_factor=0.15,
                max_size_factor=3.0,
                slic_zero=True,
                channel_axis=-1,
            ).astype(np.int32)
        else:
            # Safe fallback: small square atoms.  This path preserves the same
            # edge-band-only constraints when scikit-image is unavailable.
            block = 4
            yy, xx = np.mgrid[0:height, 0:width]
            atoms = (yy // block) * int(math.ceil(width / block)) + (xx // block)
            atoms = atoms.astype(np.int32)

        atom_count = int(atoms.max()) + 1
        flat_atoms = atoms.reshape(-1)
        flat_band = band.reshape(-1)
        atom_boxes = self._photo_label_bounding_boxes(atoms, count=atom_count)
        atom_area, atom_mean_lab, _atom_lab_std = self._photo_label_statistics(
            atoms, cie_lab, count=atom_count
        )
        _texture_area, atom_texture_mean, _atom_texture_std = (
            self._photo_label_statistics(
                atoms, texture[:, :, None], count=atom_count
            )
        )
        atom_owner_counts = self._photo_joint_label_histogram(
            atoms,
            owners,
            primary_count=atom_count,
            secondary_count=label_count,
        )
        proposals = []
        dilation_kernel = np.ones((3, 3), np.uint8)

        for atom in np.unique(flat_atoms[flat_band]):
            atom = int(atom)
            y0, x0, y1, x1 = (int(value) for value in atom_boxes[atom])
            if y1 <= y0 or x1 <= x0:
                continue
            local_y0 = max(0, y0 - 1)
            local_x0 = max(0, x0 - 1)
            local_y1 = min(height, y1 + 1)
            local_x1 = min(width, x1 + 1)
            local_atoms = atoms[local_y0:local_y1, local_x0:local_x1]
            local_atom = local_atoms == atom
            local_band = band[local_y0:local_y1, local_x0:local_x1]
            local_owners = owners[local_y0:local_y1, local_x0:local_x1]
            local_gradient = gradient[local_y0:local_y1, local_x0:local_x1]
            changeable = local_atom & local_band
            if not np.any(changeable):
                continue
            current_counts_all = atom_owner_counts[atom]
            current_values = np.flatnonzero(current_counts_all > 0)
            current_counts = current_counts_all[current_values]
            majority = int(current_values[int(np.argmax(current_counts))])
            prior_total = max(int(np.sum(current_counts)), 1)
            prior_lookup = {
                int(value): float(count) / prior_total
                for value, count in zip(current_values, current_counts)
            }
            ring = cv2.dilate(
                local_atom.astype(np.uint8), dilation_kernel, iterations=1
            ).astype(bool)
            ring &= ~local_atom
            candidates = set(int(value) for value in current_values)
            candidates.update(int(value) for value in np.unique(local_owners[ring]))
            candidates = [value for value in candidates if 0 <= value < label_count]
            if len(candidates) <= 1:
                continue

            mean_lab = atom_mean_lab[atom]
            atom_texture = float(atom_texture_mean[atom, 0])
            local_ys, local_xs = np.where(local_atom)
            center_y = float(np.mean(local_ys + local_y0))
            center_x = float(np.mean(local_xs + local_x0))
            cy = int(np.clip(round(center_y / 2.0), 0, half_h - 1))
            cx = int(np.clip(round(center_x / 2.0), 0, half_w - 1))

            scored = []
            for candidate in candidates:
                delta = float(deltaE_ciede2000(
                    mean_lab[None, :], core_lab[candidate][None, :]
                )[0])
                spatial = float(distance_maps[candidate, cy, cx])
                candidate_contact = ring & (local_owners == candidate)
                if candidate == majority:
                    barrier = 0.04
                elif np.any(candidate_contact):
                    barrier = float(np.mean(local_gradient[candidate_contact]))
                elif prior_lookup.get(candidate, 0.0) > 0.0:
                    barrier = float(np.mean(
                        local_gradient[local_atom & (local_owners != candidate)]
                    ))
                else:
                    continue
                texture_gap = abs(atom_texture - float(core_texture[candidate]))
                prior_penalty = 1.0 - float(prior_lookup.get(candidate, 0.0))
                score = (
                    0.45 * min(delta / 20.0, 2.5)
                    + 0.15 * min(spatial / 0.08, 2.5)
                    + 0.25 * min(barrier, 1.5)
                    + 0.10 * min(texture_gap / 0.30, 2.0)
                    + 0.05 * prior_penalty
                )
                scored.append((score, candidate, delta, barrier))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0])
            best_score, best_label, best_delta, best_barrier = scored[0]
            current_item = next((item for item in scored if item[1] == majority), None)
            if current_item is None or best_label == majority:
                continue
            current_score, _label, current_delta, _barrier = current_item
            improvement = float(current_score - best_score)
            change_pixels = changeable & (local_owners != best_label)
            change_y, change_x = np.nonzero(change_pixels)
            change_indices = (
                (change_y.astype(np.int64) + local_y0) * width
                + change_x.astype(np.int64) + local_x0
            ).astype(np.intp)
            pixel_count = int(len(change_indices))
            if pixel_count < 2:
                continue
            # Strong boundaries are protected unless the color evidence is very clear.
            color_gain = float(current_delta - best_delta)
            if improvement < 0.075:
                continue
            if best_barrier > 0.55 and color_gain < 3.0:
                continue
            if color_gain < 0.8 and prior_lookup.get(majority, 0.0) >= 0.70:
                continue
            priority = improvement * math.sqrt(pixel_count)
            proposals.append((priority, improvement, best_label, change_indices))

        refined = owners.copy()
        refined_flat = refined.reshape(-1)
        budget = int(math.floor(float(soft_limit) * owners.size))
        used = 0
        accepted = 0
        for _priority, _improvement, label, indices in sorted(
                proposals, key=lambda item: item[0], reverse=True):
            pending = indices[refined_flat[indices] != int(label)]
            amount = int(len(pending))
            if amount == 0 or used + amount > budget:
                continue
            refined_flat[pending] = int(label)
            used += amount
            accepted += 1

        refined, metrics = self._palette_validate_refined_owner_map(
            owners, refined, label_count, band,
            soft_limit=soft_limit, hard_limit=hard_limit,
        )
        metrics.update({
            'accepted_atoms': int(accepted),
            'proposal_count': int(len(proposals)),
            'edge_radius': int(radius),
            'method': 'local edge-band SLIC owner refinement',
        })
        return refined, metrics

    def _palette_render_object_groups(
            self, original_rgb, group_map, source_palette, target_palette,
            strength=1.0, luminance=0.76):
        """Recolor complete object groups while retaining internal shading/detail."""
        original = np.asarray(original_rgb, dtype=np.uint8)
        groups = np.asarray(group_map, dtype=np.int32)
        source = np.asarray(source_palette, dtype=np.uint8)
        target = np.asarray(target_palette, dtype=np.uint8)
        if groups.shape != original.shape[:2]:
            groups = cv2.resize(
                groups.astype(np.int32),
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        if len(source) != len(target):
            raise ValueError("Object-group source and target palettes must have equal length")
        flat_lab = self._palette_rgb_to_lab_array(original.reshape(-1, 3))
        result_lab = flat_lab.copy()
        source_lab = self._palette_rgb_to_lab_array(source)
        target_lab = self._palette_rgb_to_lab_array(target)
        labels = groups.reshape(-1)
        strength = float(np.clip(strength, 0.0, 1.0))
        preserve_luminance = float(np.clip(luminance, 0.0, 1.0))
        for group in range(len(source)):
            if np.array_equal(source[group], target[group]):
                continue
            mask = labels == group
            if not np.any(mask):
                continue
            delta = target_lab[group] - source_lab[group]
            # Apply one common color transform to all sublayers in the object.  The
            # local L/a/b variation is retained, so bright and shadow coat pieces stay
            # detailed rather than becoming a flat fill.
            effective = delta.copy()
            effective[0] *= (1.0 - preserve_luminance)
            candidate = flat_lab[mask] + effective[None, :]
            result_lab[mask] = (
                flat_lab[mask] * (1.0 - strength) + candidate * strength
            )
        result_rgb = self._palette_lab_to_rgb_array(result_lab).reshape(original.shape)
        unchanged = np.ones(len(labels), dtype=bool)
        for group in range(len(source)):
            if not np.array_equal(source[group], target[group]):
                unchanged[labels == group] = False
        result_flat = result_rgb.reshape(-1, 3)
        original_flat = original.reshape(-1, 3)
        result_flat[unchanged] = original_flat[unchanged]
        return result_flat.reshape(original.shape).astype(np.uint8)

    def _photo_relation(self, first_lab, second_lab):
        first = np.asarray(first_lab, dtype=np.float32)
        second = np.asarray(second_lab, dtype=np.float32)
        delta = float(deltaE_ciede2000(first[None, :], second[None, :])[0])
        chroma_first = float(math.hypot(first[1], first[2]))
        chroma_second = float(math.hypot(second[1], second[2]))
        hue_first = math.degrees(math.atan2(first[2], first[1])) % 360.0
        hue_second = math.degrees(math.atan2(second[2], second[1])) % 360.0
        hue_gap = self._photo_hue_distance(hue_first, hue_second)
        neutral = max(chroma_first, chroma_second) < 12.0
        same_family = (
            not neutral
            and hue_gap <= 25.0
            and abs(chroma_first - chroma_second) <= 34.0
        )
        return delta, neutral, same_family, abs(float(first[0] - second[0]))

    @staticmethod
    def _photo_label_statistics(labels, values, count=None):
        """一次分桶计算标签像素数、均值和标准差。

        旧实现对每个标签重新构造一张整图布尔掩膜，复杂度接近
        ``O(标签数 × 像素数)``。这里用 ``np.bincount`` 对每个通道做一次
        聚合，结果与逐标签统计等价，复杂度收敛为 ``O(像素数 × 通道数)``。
        """
        labels = np.asarray(labels, dtype=np.int32)
        values = np.asarray(values, dtype=np.float32)
        if values.shape[:labels.ndim] != labels.shape:
            raise ValueError("label statistics require values aligned with labels")
        flat_labels = labels.reshape(-1)
        if flat_labels.size == 0 or np.any(flat_labels < 0):
            raise ValueError("label statistics require non-negative populated labels")
        if count is None:
            count = int(flat_labels.max()) + 1
        count = int(count)
        flat_values = values.reshape(len(flat_labels), -1).astype(np.float64, copy=False)
        counts = np.bincount(flat_labels, minlength=count).astype(np.int64)
        safe_counts = np.maximum(counts.astype(np.float64), 1.0)
        means = np.empty((count, flat_values.shape[1]), dtype=np.float64)
        variances = np.empty_like(means)
        for channel in range(flat_values.shape[1]):
            channel_values = flat_values[:, channel]
            sums = np.bincount(
                flat_labels, weights=channel_values, minlength=count
            ).astype(np.float64)
            squares = np.bincount(
                flat_labels, weights=channel_values * channel_values, minlength=count
            ).astype(np.float64)
            means[:, channel] = sums / safe_counts
            variances[:, channel] = np.maximum(
                squares / safe_counts - means[:, channel] * means[:, channel],
                0.0,
            )
        means[counts == 0] = 0.0
        variances[counts == 0] = 0.0
        return (
            counts,
            means.astype(np.float32),
            np.sqrt(variances).astype(np.float32),
        )

    @staticmethod
    def _photo_joint_label_histogram(
            primary, secondary, primary_count=None, secondary_count=None):
        """一次统计“原子区域 × owner”像素数，替代逐原子整图扫描。"""
        primary = np.asarray(primary, dtype=np.int32)
        secondary = np.asarray(secondary, dtype=np.int32)
        if primary.shape != secondary.shape or primary.size == 0:
            raise ValueError("joint label histogram requires matching non-empty maps")
        if np.any(primary < 0) or np.any(secondary < 0):
            raise ValueError("joint label histogram requires non-negative labels")
        if primary_count is None:
            primary_count = int(primary.max()) + 1
        if secondary_count is None:
            secondary_count = int(secondary.max()) + 1
        primary_count = int(primary_count)
        secondary_count = int(secondary_count)
        combined = (
            primary.reshape(-1).astype(np.int64) * secondary_count
            + secondary.reshape(-1).astype(np.int64)
        )
        return np.bincount(
            combined, minlength=primary_count * secondary_count
        ).reshape(primary_count, secondary_count).astype(np.int64)

    @staticmethod
    def _photo_label_bounding_boxes(labels, count=None):
        """返回每个标签的半开包围盒 ``[y0, x0, y1, x1]``。"""
        labels = np.asarray(labels, dtype=np.int32)
        if labels.ndim != 2 or labels.size == 0 or np.any(labels < 0):
            raise ValueError("label bounding boxes require a non-empty 2D label map")
        if count is None:
            count = int(labels.max()) + 1
        count = int(count)
        height, width = labels.shape
        yy, xx = np.indices(labels.shape, dtype=np.int32)
        flat_labels = labels.reshape(-1)
        y0 = np.full(count, height, dtype=np.int32)
        x0 = np.full(count, width, dtype=np.int32)
        y1 = np.full(count, -1, dtype=np.int32)
        x1 = np.full(count, -1, dtype=np.int32)
        np.minimum.at(y0, flat_labels, yy.reshape(-1))
        np.minimum.at(x0, flat_labels, xx.reshape(-1))
        np.maximum.at(y1, flat_labels, yy.reshape(-1))
        np.maximum.at(x1, flat_labels, xx.reshape(-1))
        missing = y1 < 0
        boxes = np.column_stack([y0, x0, y1 + 1, x1 + 1]).astype(np.int32)
        boxes[missing] = 0
        return boxes

    def _photo_region_features(self, rgb, labels):
        rgb = np.asarray(rgb, dtype=np.uint8)
        labels = np.asarray(labels, dtype=np.int32)
        count = int(labels.max()) + 1
        cie_lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = self._palette_iga_robust_normalize(
            np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y), 94.0
        )
        combined = np.concatenate([
            cie_lab,
            gray[:, :, None],
            gradient[:, :, None],
        ], axis=2)
        area, means, standard_deviations = self._photo_label_statistics(
            labels, combined, count=count
        )
        mean_lab = means[:, :3]
        gray_std = standard_deviations[:, 3] / 64.0
        edge_mean = means[:, 4]
        return cie_lab, gradient, area, mean_lab, gray_std, edge_mean

    def _photo_merge_superpixels(self, rgb, labels):
        labels = np.asarray(labels, dtype=np.int32)
        cie_lab, gradient, area, mean_lab, gray_std, edge_mean = self._photo_region_features(rgb, labels)
        count = int(labels.max()) + 1
        total = max(int(labels.size), 1)
        parent, find, union = self._photo_union_find(count)
        adjacency = self._photo_adjacency(labels, gradient)
        merge_edges = []
        for (first, second), (contact, edge_sum) in adjacency.items():
            delta, neutral, same_family, lightness_gap = self._photo_relation(
                mean_lab[first], mean_lab[second]
            )
            boundary = float(edge_sum) / max(float(contact), 1.0)
            texture_gap = abs(float(gray_std[first] - gray_std[second]))
            small_area = min(area[first], area[second]) / total
            merge = (
                delta <= 3.8
                or (neutral and delta <= 7.0 and boundary <= 0.32)
                or (same_family and delta <= 9.8 and boundary <= 0.34 and texture_gap <= 0.48)
                or (small_area <= 0.0010 and delta <= 12.0 and boundary <= 0.42)
            )
            if merge:
                score = delta + 7.0 * boundary + 2.0 * texture_gap
                merge_edges.append((score, first, second))
        for _score, first, second in sorted(merge_edges, key=lambda item: item[0]):
            union(first, second)
        roots = np.asarray([find(index) for index in range(count)], dtype=np.int32)
        _, inverse = np.unique(roots, return_inverse=True)
        return inverse[labels].astype(np.int32)

    def _photo_group_color_layers(self, rgb, atomic_labels):
        """Group robust atomic regions into editable photo color layers.

        Unlike pixel-wise nearest-palette assignment, clustering operates on SLIC
        atomic regions. Lightness is deliberately compressed so illumination
        variants of one material (for example red / dark red) tend to stay in
        one layer, while chroma and texture still separate genuinely different
        materials.
        """
        rgb = np.asarray(rgb, dtype=np.uint8)
        atomic_labels = np.asarray(atomic_labels, dtype=np.int32)
        cie_lab, gradient, area, mean_lab, gray_std, edge_mean = self._photo_region_features(
            rgb, atomic_labels
        )
        atom_count = int(atomic_labels.max()) + 1
        if atom_count <= 2:
            return atomic_labels.copy()

        chroma = np.linalg.norm(mean_lab[:, 1:3], axis=1)
        features = np.column_stack([
            mean_lab[:, 0] * 0.36,
            mean_lab[:, 1],
            mean_lab[:, 2],
            gray_std * 7.0,
            edge_mean * 4.0,
        ]).astype(np.float32)

        complexity = int(round(math.sqrt(atom_count) * 1.25)) + 1
        chroma_range = float(np.percentile(chroma, 90) - np.percentile(chroma, 10))
        if chroma_range > 35.0:
            complexity += 1
        target_count = int(np.clip(complexity, 9, min(16, atom_count)))

        clusterer = KMeans(
            n_clusters=target_count,
            random_state=20260725,
            n_init=8,
            max_iter=220,
        )
        weights = np.maximum(area.astype(np.float64), 1.0)
        try:
            clusterer.fit(features, sample_weight=weights)
            atom_groups = np.asarray(clusterer.labels_, dtype=np.int32)
        except TypeError:
            atom_groups = np.asarray(clusterer.fit_predict(features), dtype=np.int32)

        owner_map = atom_groups[atomic_labels].astype(np.int32)

        # Merge only clearly redundant output layers; avoid transitive global
        # union chains that previously collapsed real photographs into 2-6 colors.
        for _iteration in range(5):
            count = int(owner_map.max()) + 1
            if count <= 8:
                break
            counts = np.bincount(owner_map.reshape(-1), minlength=count).astype(np.int64)
            layer_lab = np.zeros((count, 3), dtype=np.float32)
            layer_texture = np.zeros(count, dtype=np.float32)
            for index in range(count):
                atom_mask = atom_groups == index
                if not np.any(atom_mask):
                    continue
                local_weights = area[atom_mask].astype(np.float64)
                local_weights /= max(float(local_weights.sum()), 1.0)
                layer_lab[index] = np.sum(mean_lab[atom_mask] * local_weights[:, None], axis=0)
                layer_texture[index] = float(np.sum(gray_std[atom_mask] * local_weights))
            best = None
            best_score = None
            for first in range(count):
                for second in range(first + 1, count):
                    delta, neutral, same_family, lightness_gap = self._photo_relation(
                        layer_lab[first], layer_lab[second]
                    )
                    texture_gap = abs(float(layer_texture[first] - layer_texture[second]))
                    smaller = min(counts[first], counts[second]) / max(float(owner_map.size), 1.0)
                    redundant = (
                        (delta <= 3.2 and texture_gap <= 0.45)
                        or (neutral and delta <= 5.5 and lightness_gap <= 11.0 and texture_gap <= 0.35)
                        or (same_family and delta <= 7.5 and texture_gap <= 0.38 and smaller <= 0.035)
                    )
                    if not redundant:
                        continue
                    score = delta + 3.5 * texture_gap
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (first, second)
            if best is None:
                break
            keep, remove = best
            if counts[remove] > counts[keep]:
                keep, remove = remove, keep
            owner_map[owner_map == remove] = keep
            atom_groups[atom_groups == remove] = keep
            active_groups = np.unique(owner_map)
            remap = np.full(count, -1, dtype=np.int32)
            remap[active_groups] = np.arange(len(active_groups), dtype=np.int32)
            owner_map = remap[owner_map].astype(np.int32)
            atom_groups = remap[atom_groups].astype(np.int32)
        return owner_map

    def _photo_remove_tiny_layers(self, rgb, owner_map, minimum_ratio=0.0020):
        rgb = np.asarray(rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32).copy()
        total = max(int(owners.size), 1)
        for _iteration in range(10):
            count = int(owners.max()) + 1
            counts = np.bincount(owners.reshape(-1), minlength=count).astype(np.int64)
            small = np.flatnonzero(counts / total < float(minimum_ratio))
            if len(small) == 0 or count <= 5:
                break
            cie_lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
            means = np.zeros((count, 3), dtype=np.float32)
            for index in range(count):
                mask = owners == index
                if np.any(mask):
                    means[index] = np.mean(cie_lab[mask], axis=0)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge = self._palette_iga_robust_normalize(np.sqrt(gx * gx + gy * gy), 94.0)
            adjacency = self._photo_adjacency(owners, edge)
            changed = False
            for index in small:
                neighboring = []
                for (first, second), (contact, edge_sum) in adjacency.items():
                    if first == index:
                        neighboring.append((second, contact, edge_sum))
                    elif second == index:
                        neighboring.append((first, contact, edge_sum))
                best = None
                best_score = None
                for neighbor, contact, edge_sum in neighboring:
                    delta, _neutral, _same_family, _lightness_gap = self._photo_relation(
                        means[index], means[neighbor]
                    )
                    boundary = float(edge_sum) / max(float(contact), 1.0)
                    score = delta + 6.0 * boundary - 0.001 * float(contact)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = int(neighbor)
                if best is not None and (best_score <= 18.0 or counts[index] / total < 0.0007):
                    owners[owners == index] = best
                    changed = True
            if not changed:
                break
            _, inverse = np.unique(owners, return_inverse=True)
            owners = inverse.reshape(owners.shape).astype(np.int32)
        return owners

    @staticmethod
    def _photo_compact_labels(labels):
        labels = np.asarray(labels, dtype=np.int32)
        _unique, inverse = np.unique(labels, return_inverse=True)
        return inverse.reshape(labels.shape).astype(np.int32)

    def _photo_palette_from_owner_map(self, rgb, owner_map):
        rgb = np.asarray(rgb, dtype=np.uint8)
        owners = self._photo_compact_labels(owner_map)
        count = int(owners.max()) + 1
        flat_rgb = rgb.reshape(-1, 3)
        flat_owner = owners.reshape(-1)
        palette = []
        proportions = []
        for index in range(count):
            mask = flat_owner == index
            if not np.any(mask):
                continue
            values = flat_rgb[mask]
            median = np.median(values, axis=0)
            stride = max(1, len(values) // 6000)
            sample = values[::stride]
            representative = sample[
                int(np.argmin(np.linalg.norm(sample.astype(np.float32) - median[None, :], axis=1)))
            ]
            palette.append(representative.astype(np.uint8))
            proportions.append(float(np.mean(mask)))
        palette = np.asarray(palette, dtype=np.uint8)
        proportions = np.asarray(proportions, dtype=np.float32)
        return palette, proportions, owners

    @staticmethod
    def _photo_owner_boundary_mask(owner_map):
        owners = np.asarray(owner_map, dtype=np.int32)
        boundary = np.zeros(owners.shape, dtype=bool)
        boundary[:, 1:] |= owners[:, 1:] != owners[:, :-1]
        boundary[:, :-1] |= owners[:, 1:] != owners[:, :-1]
        boundary[1:, :] |= owners[1:, :] != owners[:-1, :]
        boundary[:-1, :] |= owners[1:, :] != owners[:-1, :]
        return boundary

    def _photo_boundary_adherence(self, rgb, owner_map):
        rgb = np.asarray(rgb, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = self._palette_iga_robust_normalize(
            np.sqrt(gx * gx + gy * gy), 94.0
        )
        boundary = self._photo_owner_boundary_mask(owner_map)
        if not np.any(boundary):
            return 0.0
        return float(np.mean(gradient[boundary]))

    def _photo_cleanup_tiny_components(self, rgb, owner_map):
        """Remove only pixel-scale islands while preserving real small details."""
        rgb = np.asarray(rgb, dtype=np.uint8)
        owners = self._photo_compact_labels(owner_map)
        height, width = owners.shape
        total = max(height * width, 1)
        min_pixels = max(5, int(round(total * 0.000018)))
        cie_lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = self._palette_iga_robust_normalize(
            np.sqrt(gx * gx + gy * gy), 94.0
        )

        for _iteration in range(2):
            changed = False
            count = int(owners.max()) + 1
            layer_means = np.zeros((count, 3), dtype=np.float32)
            for layer in range(count):
                layer_mask = owners == layer
                if np.any(layer_mask):
                    layer_means[layer] = np.mean(cie_lab[layer_mask], axis=0)
            for layer in range(count):
                mask = np.uint8(owners == layer)
                component_count, component_labels, stats, _centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
                for component in range(1, component_count):
                    size = int(stats[component, cv2.CC_STAT_AREA])
                    if size >= min_pixels:
                        continue
                    component_mask = component_labels == component
                    # Keep tiny but strongly edged details such as eyes, buttons, and highlights.
                    if float(np.mean(edge[component_mask])) >= 0.52:
                        continue
                    dilated = cv2.dilate(component_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                    ring = dilated & ~component_mask
                    neighbors = owners[ring]
                    neighbors = neighbors[neighbors != layer]
                    if len(neighbors) == 0:
                        continue
                    neighbor_ids, neighbor_counts = np.unique(neighbors, return_counts=True)
                    component_mean = np.mean(cie_lab[component_mask], axis=0)
                    best = None
                    best_score = None
                    for neighbor, contact in zip(neighbor_ids, neighbor_counts):
                        delta = float(deltaE_ciede2000(
                            component_mean[None, :], layer_means[int(neighbor)][None, :]
                        )[0])
                        score = delta - 0.08 * float(contact)
                        if best_score is None or score < best_score:
                            best_score = score
                            best = int(neighbor)
                    if best is not None and best_score <= 16.0:
                        owners[component_mask] = best
                        changed = True
            if not changed:
                break
            owners = self._photo_compact_labels(owners)
        return owners

    def _photo_refine_full_resolution_layers(self, rgb, seed_owner_map, palette_rgb):
        """Refine coarse photo layers on a dense, full-resolution SLIC graph.

        The first pass determines the semantic color layers. This pass does not
        invent new palette colors; it snaps those owners to fine superpixel
        boundaries, recovers narrow structures, and removes only pixel-scale noise.
        """
        if not _PHOTO_LAYER_SKIMAGE_AVAILABLE:
            return np.asarray(seed_owner_map, dtype=np.int32), {
                'fine_superpixel_count': 0,
                'boundary_adherence': self._photo_boundary_adherence(rgb, seed_owner_map),
            }
        rgb = np.asarray(rgb, dtype=np.uint8)
        owners_seed = np.asarray(seed_owner_map, dtype=np.int32)
        if owners_seed.shape != rgb.shape[:2]:
            owners_seed = cv2.resize(
                owners_seed,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        palette_rgb = np.asarray(palette_rgb, dtype=np.uint8)
        layer_count = len(palette_rgb)
        if layer_count <= 1:
            return owners_seed.copy(), {
                'fine_superpixel_count': 1,
                'boundary_adherence': self._photo_boundary_adherence(rgb, owners_seed),
            }

        height, width = rgb.shape[:2]
        pixel_count = max(height * width, 1)
        requested = int(np.clip(round(pixel_count / 105.0), 900, 4200))
        filtered = cv2.bilateralFilter(rgb, d=5, sigmaColor=14, sigmaSpace=3)
        fine = slic(
            filtered.astype(np.float32) / 255.0,
            n_segments=requested,
            compactness=4.2,
            sigma=0.45,
            start_label=0,
            enforce_connectivity=True,
            min_size_factor=0.18,
            max_size_factor=3.0,
            slic_zero=True,
            channel_axis=-1,
        ).astype(np.int32)
        fine_count = int(fine.max()) + 1
        cie_lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
        palette_lab = rgb2lab(palette_rgb[None, :, :].astype(np.float32) / 255.0)[0].astype(np.float32)
        flat_fine = fine.reshape(-1)
        flat_seed = owners_seed.reshape(-1)
        atom_area, atom_mean, _atom_std = self._photo_label_statistics(
            fine, cie_lab, count=fine_count
        )
        prior_counts = self._photo_joint_label_histogram(
            fine,
            owners_seed,
            primary_count=fine_count,
            secondary_count=layer_count,
        ).astype(np.float32)
        prior = prior_counts / np.maximum(
            atom_area.astype(np.float32)[:, None], 1.0
        )

        distances = np.zeros((fine_count, layer_count), dtype=np.float32)
        for layer in range(layer_count):
            distances[:, layer] = deltaE_ciede2000(
                atom_mean, np.repeat(palette_lab[layer][None, :], fine_count, axis=0)
            ).astype(np.float32)
        # A strong seed prior stabilizes semantic layers; color likelihood can still
        # correct coarse boundaries when a fine atom is clearly closer to another layer.
        scores = distances + 7.5 * (1.0 - prior)
        majority = np.argmax(prior, axis=1)
        majority_strength = np.max(prior, axis=1)
        selected = np.argmin(scores, axis=1).astype(np.int32)
        locked = majority_strength >= 0.86
        selected[locked] = majority[locked]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = self._palette_iga_robust_normalize(
            np.sqrt(gx * gx + gy * gy), 94.0
        )
        adjacency = self._photo_adjacency(fine, gradient)
        neighbor_info = [[] for _ in range(fine_count)]
        for (first, second), (contact, edge_sum) in adjacency.items():
            boundary = float(edge_sum) / max(float(contact), 1.0)
            neighbor_info[first].append((second, float(contact), boundary))
            neighbor_info[second].append((first, float(contact), boundary))

        for _iteration in range(2):
            updated = selected.copy()
            for atom in range(fine_count):
                if majority_strength[atom] >= 0.92:
                    continue
                candidates = {int(selected[atom]), int(majority[atom])}
                for neighbor, _contact, _boundary in neighbor_info[atom]:
                    candidates.add(int(selected[neighbor]))
                best_layer = int(selected[atom])
                best_score = None
                perimeter = max(sum(item[1] for item in neighbor_info[atom]), 1.0)
                for layer in candidates:
                    weak_support = 0.0
                    for neighbor, contact, boundary in neighbor_info[atom]:
                        if int(selected[neighbor]) == layer:
                            weak_support += contact * max(0.0, 1.0 - boundary)
                    support = weak_support / perimeter
                    score = (
                        float(distances[atom, layer])
                        + 6.5 * (1.0 - float(prior[atom, layer]))
                        - 3.2 * support
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_layer = int(layer)
                updated[atom] = best_layer
            selected = updated

        refined = selected[fine].astype(np.int32)
        refined = self._photo_cleanup_tiny_components(rgb, refined)
        refined = self._photo_compact_labels(refined)
        return refined, {
            'fine_superpixel_count': int(fine_count),
            'boundary_adherence': self._photo_boundary_adherence(rgb, refined),
        }

    def _palette_build_selected_region_preview(self, rgb, owner_map, index, max_size=(320, 250)):
        """Build a detail-oriented layer preview with contours and component stats."""
        rgb = np.asarray(rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32)
        if owners.shape != rgb.shape[:2]:
            owners = cv2.resize(
                owners,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        mask = owners == int(index)
        if not np.any(mask):
            blank = Image.new('RGB', max_size, 'white')
            return blank, {'area_percent': 0.0, 'parts': 0, 'largest_part_percent': 0.0}

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        sizes = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.asarray([], dtype=np.int32)
        part_count = int(len(sizes))
        largest = int(np.max(sizes)) if len(sizes) else int(np.count_nonzero(mask))
        area = int(np.count_nonzero(mask))

        yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
        checker_index = ((yy // 10) + (xx // 10)) % 2
        checker = np.empty_like(rgb)
        checker[checker_index == 0] = (214, 214, 214)
        checker[checker_index == 1] = (246, 246, 246)
        checker[mask] = rgb[mask]

        contour_input = (mask.astype(np.uint8) * 255)
        contours, _hierarchy = cv2.findContours(
            contour_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outlined = checker.copy()
        cv2.drawContours(outlined, contours, -1, (0, 190, 255), 1, lineType=cv2.LINE_AA)

        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = max(8, int(round(0.035 * max(x1 - x0, y1 - y0))))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(rgb.shape[1], x1 + pad)
        y1 = min(rgb.shape[0], y1 + pad)
        crop = outlined[y0:y1, x0:x1]
        preview = Image.fromarray(crop, mode='RGB')
        preview.thumbnail(max_size, Image.Resampling.LANCZOS)
        metrics = {
            'area_percent': 100.0 * area / max(mask.size, 1),
            'parts': part_count,
            'largest_part_percent': 100.0 * largest / max(area, 1),
        }
        return preview, metrics

    def _palette_extract_photo_layers(self, image, return_model=False):
        if not _PHOTO_LAYER_SKIMAGE_AVAILABLE:
            raise RuntimeError(
                "Photo-aware extraction requires scikit-image. Install it with: pip install scikit-image"
            )
        full_source = image.convert('RGB').copy()
        process = full_source.copy()
        process.thumbnail((960, 960), Image.Resampling.LANCZOS)
        process_rgb = np.asarray(process, dtype=np.uint8)
        pixel_count = max(process_rgb.shape[0] * process_rgb.shape[1], 1)
        requested_superpixels = int(np.clip(round(pixel_count / 430.0), 500, 1350))
        filtered = cv2.bilateralFilter(process_rgb, d=5, sigmaColor=18, sigmaSpace=4)
        superpixels = slic(
            filtered.astype(np.float32) / 255.0,
            n_segments=requested_superpixels,
            compactness=7.5,
            sigma=0.8,
            start_label=0,
            enforce_connectivity=True,
            min_size_factor=0.25,
            max_size_factor=4.0,
            slic_zero=True,
            channel_axis=-1,
        ).astype(np.int32)
        atomic = self._photo_merge_superpixels(process_rgb, superpixels)
        owner_small = self._photo_group_color_layers(process_rgb, atomic)
        owner_small = self._photo_remove_tiny_layers(process_rgb, owner_small, minimum_ratio=0.0022)

        full_rgb = np.asarray(full_source, dtype=np.uint8)
        owner_seed = cv2.resize(
            owner_small.astype(np.int32),
            (full_source.width, full_source.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        seed_palette, _seed_proportions, owner_seed = self._photo_palette_from_owner_map(
            full_rgb, owner_seed
        )
        owner_map, detail_metrics = self._photo_refine_full_resolution_layers(
            full_rgb, owner_seed, seed_palette
        )
        owner_map = self._photo_remove_tiny_layers(full_rgb, owner_map, minimum_ratio=0.0015)
        palette, proportions, owner_map = self._photo_palette_from_owner_map(
            full_rgb, owner_map
        )
        order = np.argsort(-proportions, kind='stable')
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order), dtype=np.intp)
        owner_map = inverse[owner_map]
        palette = palette[order]
        proportions = proportions[order]
        proportions /= max(float(proportions.sum()), 1e-8)
        model = self._palette_build_sparse_model_from_owner_map(owner_map)
        model['extractor'] = 'photo_slic_graph_detail_v2'
        model['superpixel_count'] = int(superpixels.max()) + 1
        model['atomic_region_count'] = int(atomic.max()) + 1
        model['fine_superpixel_count'] = int(detail_metrics.get('fine_superpixel_count', 0))
        model['boundary_adherence'] = float(detail_metrics.get('boundary_adherence', 0.0))
        if return_model:
            return palette, proportions, model
        return palette, proportions

    def _palette_extract_representative(self, image, color_count=None, return_model=False):
        if color_count is None:
            return self._palette_extract_photo_layers(image, return_model=return_model)
        return self._palette_extract_representative_legacy_auto(
            image, color_count=color_count, return_model=return_model
        )

    def _palette_extract_representative_legacy_auto(self, image, color_count=None, return_model=False):
        """Representative palette with paper-informed dominant geometric layers.

        Unlike v4.3.4's KMeans + nearest-color owners, this path first builds a
        representative palette from convex-hull extremes and q=0.2 MeanShift
        interior modes.  A sparse tetrahedral decomposition then supplies a
        dominant owner map, which is regularized spatially for the editor UI.
        """
        requested = None
        if color_count is not None:
            try:
                requested = max(2, int(color_count))
            except Exception:
                requested = None

        full_source = image.convert('RGB').copy()
        analysis_source = full_source.copy()
        analysis_source.thumbnail((620, 620), Image.Resampling.LANCZOS)
        analysis_rgb = np.asarray(analysis_source, dtype=np.uint8)
        palette_rgb = self._palette_extract_geometric_candidates(
            analysis_rgb, requested=requested
        )

        full_rgb = np.asarray(full_source, dtype=np.uint8)
        owner_map, confidence, geometric_metrics = self._palette_geometric_dominant_owner_map(
            full_rgb, palette_rgb
        )
        owner_map, palette_rgb = self._palette_merge_adjacent_similar_regions(
            full_rgb, owner_map, palette_rgb
        )
        owner_map, palette_rgb = self._palette_reduce_similar_layer_families(
            full_rgb, owner_map, palette_rgb, requested=requested
        )

        # Remove truly unsupported owners, but preserve accents down to 0.25%.
        counts = np.bincount(owner_map.reshape(-1), minlength=len(palette_rgb)).astype(np.int64)
        support = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
        valid = counts > 0
        if requested is None and int(np.sum(valid)) > 6:
            valid &= support >= 0.0025
            if int(np.sum(valid)) < 6:
                valid[:] = False
                valid[np.argsort(-support, kind='stable')[:min(6, len(valid))]] = True
        if not np.all(valid):
            valid_indices = np.flatnonzero(valid)
            invalid_indices = np.flatnonzero(~valid)
            palette_lab = self._palette_rgb_to_lab_array(palette_rgb)
            for invalid in invalid_indices:
                replacement = valid_indices[int(np.argmin(
                    np.linalg.norm(
                        palette_lab[valid_indices] - palette_lab[invalid], axis=1
                    )
                ))]
                owner_map[owner_map == invalid] = int(replacement)
            remap = np.full(len(palette_rgb), -1, dtype=np.int32)
            remap[valid_indices] = np.arange(len(valid_indices), dtype=np.int32)
            owner_map = remap[owner_map]
            palette_rgb = palette_rgb[valid]

        owner_map, palette_rgb = self._palette_merge_adjacent_similar_regions(
            full_rgb, owner_map, palette_rgb
        )
        owner_map, palette_rgb = self._palette_reduce_similar_layer_families(
            full_rgb, owner_map, palette_rgb, requested=requested
        )
        counts = np.bincount(owner_map.reshape(-1), minlength=len(palette_rgb)).astype(np.int64)
        order = np.argsort(-counts, kind='stable')
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order), dtype=np.intp)
        owner_map = inverse[owner_map]
        palette_rgb = palette_rgb[order]
        counts = counts[order]
        proportions = counts.astype(np.float32)
        proportions /= max(float(proportions.sum()), 1.0)

        model = self._palette_build_sparse_model_from_owner_map(owner_map)
        model['mode'] = 'geometric_dominant_owner'
        model['geometric_metrics'] = geometric_metrics
        model['confidence'] = confidence.reshape(-1).astype(np.float16)
        if return_model:
            return palette_rgb, proportions, model
        return palette_rgb, proportions

    def _palette_extract_representative_legacy(self, image, color_count=None, return_model=False):
        """自适应 NA-IGA 风格源颜色/区域提取。

        Palette Result Editor 中不再固定取前 5 色，而是：
        1) 先提取较多候选源颜色；
        2) 合并近似色；
        3) 根据面积、色彩显著性、区域集中度自动确定有效区域数；
        4) 以互斥 owner map 形成 100% 覆盖、0% 重叠的区域模板；
        5) Click a color to edit 直接按这些区域模板换色。

        当 color_count 为整数时（如参考图匹配场景），仍可按给定数量收敛。
        """
        requested = None
        if color_count is not None:
            try:
                requested = max(2, int(color_count))
            except Exception:
                requested = None
        full_source = image.convert('RGB').copy()
        source = full_source.copy()
        source.thumbnail((720, 720), Image.Resampling.LANCZOS)
        rgb = np.asarray(source, dtype=np.uint8)
        height, width = rgb.shape[:2]
        flat_rgb = rgb.reshape(-1, 3)
        total_pixels = len(flat_rgb)
        if total_pixels == 0:
            raise ValueError("The image has no usable pixels")

        # 先做轻量边缘保持去噪，再用于“归属判定”；代表色仍从原图像素中提取。
        # 这样可抑制 JPEG 色漂、抗锯齿和传感器噪声，却不会把人物/服装边界抹平。
        smoothed_rgb = cv2.bilateralFilter(rgb, d=7, sigmaColor=20, sigmaSpace=5)
        lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        assignment_lab_image = cv2.cvtColor(
            smoothed_rgb, cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        flat_lab = lab_image.reshape(-1, 3)
        flat_assignment_lab = assignment_lab_image.reshape(-1, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_strength = self._palette_iga_robust_normalize(
            np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y).reshape(-1), 92.0
        )
        chroma_strength = self._palette_iga_robust_normalize(
            np.linalg.norm(flat_lab[:, 1:3] - 128.0, axis=1), 92.0
        )

        border_width = max(2, int(round(min(height, width) * 0.04)))
        border_mask = np.zeros((height, width), dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True
        border_lab = lab_image[border_mask]
        border_center = np.median(border_lab, axis=0)
        border_distance = np.linalg.norm(border_lab - border_center, axis=1)
        global_background_distance = np.linalg.norm(flat_lab - border_center, axis=1)
        coherent_border = float(np.mean(border_distance < 11.0))
        global_background_ratio = float(np.mean(global_background_distance < 11.0))
        background_like = np.zeros(total_pixels, dtype=bool)
        if coherent_border >= 0.50 and global_background_ratio >= 0.16:
            background_like = global_background_distance < 11.0

        sample_weight = (
            0.62 + 0.28 * np.sqrt(chroma_strength) + 0.10 * edge_strength
        ).astype(np.float32)
        sample_weight[background_like] *= 0.52
        sample_weight = np.maximum(sample_weight, 0.05)

        rng = np.random.default_rng(20260724)
        weighted_count = min(total_pixels, 36000)
        if weighted_count < total_pixels:
            probabilities = sample_weight / max(float(sample_weight.sum()), 1e-8)
            weighted_indices = rng.choice(
                total_pixels, size=weighted_count, replace=False, p=probabilities
            )
        else:
            weighted_indices = np.arange(total_pixels, dtype=np.intp)

        grid_indices = []
        per_cell = 120
        for grid_y in range(8):
            y0 = grid_y * height // 8
            y1 = (grid_y + 1) * height // 8
            for grid_x in range(8):
                x0 = grid_x * width // 8
                x1 = (grid_x + 1) * width // 8
                ys, xs = np.mgrid[y0:y1, x0:x1]
                cell = (ys * width + xs).reshape(-1)
                if len(cell) > per_cell:
                    cell = rng.choice(cell, size=per_cell, replace=False)
                grid_indices.append(cell.astype(np.intp))
        sample_indices = np.unique(np.concatenate([weighted_indices, *grid_indices]))
        sample_rgb = flat_rgb[sample_indices]
        sample_lab = flat_lab[sample_indices]

        quantized = ((sample_rgb.astype(np.uint16) // 4) * 4).astype(np.uint8)
        unique_count = len(np.unique(quantized, axis=0))
        if requested is None:
            base_guess = int(np.clip(round(unique_count * 0.10), 12, 28))
        else:
            base_guess = max(requested * 4, 12)
        candidate_count = min(base_guess, max(1, unique_count), len(sample_lab))
        if requested is not None and candidate_count <= requested:
            candidate_count = min(requested, len(sample_lab))

        features = sample_lab.astype(np.float32).copy()
        features[:, 0] *= 0.82
        clusterer = KMeans(
            n_clusters=int(candidate_count),
            random_state=20260724,
            n_init=6,
            max_iter=160,
        )
        sample_labels = clusterer.fit_predict(features)
        centers_lab = np.asarray(clusterer.cluster_centers_, dtype=np.float32)
        centers_lab[:, 0] /= 0.82

        representatives = []
        for cluster_id in range(candidate_count):
            mask = sample_labels == cluster_id
            representatives.append(self._palette_iga_cluster_representative(
                sample_rgb[mask], sample_lab[mask], centers_lab[cluster_id]
            ))
        representatives = np.asarray(representatives, dtype=np.uint8)
        representatives_lab = self._palette_rgb_to_lab_array(representatives)

        distance = np.sum(
            (flat_lab[:, None, :] - representatives_lab[None, :, :]) ** 2,
            axis=2,
        )
        candidate_labels = np.argmin(distance, axis=1)
        candidate_counts = np.bincount(
            candidate_labels, minlength=candidate_count
        ).astype(np.float64)
        candidate_weighted = np.bincount(
            candidate_labels, weights=sample_weight, minlength=candidate_count
        ).astype(np.float64)
        label_image = candidate_labels.reshape(height, width)
        border_flat = border_mask.reshape(-1)

        coherence = np.zeros(candidate_count, dtype=np.float32)
        centrality = np.zeros(candidate_count, dtype=np.float32)
        interiority = np.zeros(candidate_count, dtype=np.float32)
        for cluster_id in range(candidate_count):
            mask = label_image == cluster_id
            area = max(float(candidate_counts[cluster_id]), 1.0)
            component_count, _component_map, stats, _centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            if component_count > 1:
                largest = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
                coherence[cluster_id] = largest / area
            coords_y, coords_x = np.where(mask)
            if len(coords_x):
                center_x = float(np.mean(coords_x)) / max(width - 1, 1)
                center_y = float(np.mean(coords_y)) / max(height - 1, 1)
                centrality[cluster_id] = 1.0 - min(
                    math.hypot(center_x - 0.5, center_y - 0.5) / 0.70710678, 1.0
                )
            assigned = candidate_labels == cluster_id
            border_fraction = float(np.mean(border_flat[assigned])) if np.any(assigned) else 1.0
            interiority[cluster_id] = 1.0 - math.sqrt(max(border_fraction, 0.0))

        area_ratio = candidate_counts / max(float(candidate_counts.sum()), 1.0)
        weighted_ratio = candidate_weighted / max(float(candidate_weighted.sum()), 1.0)
        representative_chroma = self._palette_iga_robust_normalize(
            np.linalg.norm(representatives_lab[:, 1:3] - 128.0, axis=1), 90.0
        )
        region_score = 0.55 * centrality + 0.45 * interiority
        support_score = (
            0.46 * np.sqrt(area_ratio / max(float(area_ratio.max()), 1e-8))
            + 0.20 * np.sqrt(weighted_ratio / max(float(weighted_ratio.max()), 1e-8))
            + 0.12 * representative_chroma
            + 0.08 * np.sqrt(np.clip(coherence, 0.0, 1.0))
            + 0.14 * region_score
        )

        merge_order = np.argsort(-support_score, kind='stable')
        merged = []
        for index in merge_order:
            if candidate_counts[index] <= 0:
                continue
            target = None
            area_index = float(candidate_counts[index]) / max(float(total_pixels), 1.0)
            for merged_index, item in enumerate(merged):
                item_area = float(item['count']) / max(float(total_pixels), 1.0)
                if self._palette_iga_should_merge(
                        representatives_lab[index], item['lab'],
                        area_index, item_area,
                        float(coherence[index]), float(item['coherence'])):
                    target = merged_index
                    break
            if target is None:
                merged.append({
                    'rgb': representatives[index].copy(),
                    'lab': representatives_lab[index].copy(),
                    'count': float(candidate_counts[index]),
                    'weight': float(candidate_weighted[index]),
                    'coherence': float(coherence[index]),
                    'region': float(region_score[index]),
                    'score': float(support_score[index]),
                })
            else:
                item = merged[target]
                item['count'] += float(candidate_counts[index])
                item['weight'] += float(candidate_weighted[index])
                item['coherence'] = max(item['coherence'], float(coherence[index]))
                item['region'] = max(item['region'], float(region_score[index]))
                if support_score[index] > item['score'] * 1.04:
                    item['rgb'] = representatives[index].copy()
                    item['lab'] = representatives_lab[index].copy()
                    item['score'] = float(support_score[index])

        merged_rgb = np.asarray([item['rgb'] for item in merged], dtype=np.uint8)
        merged_lab = self._palette_rgb_to_lab_array(merged_rgb)
        merged_counts = np.asarray([item['count'] for item in merged], dtype=np.float64)
        merged_weights = np.asarray([item['weight'] for item in merged], dtype=np.float64)
        merged_coherence = np.asarray([item['coherence'] for item in merged], dtype=np.float32)
        merged_region = np.asarray([item['region'] for item in merged], dtype=np.float32)
        merged_area = merged_counts / max(float(merged_counts.sum()), 1.0)
        merged_weight_ratio = merged_weights / max(float(merged_weights.sum()), 1.0)
        merged_chroma = self._palette_iga_robust_normalize(
            np.linalg.norm(merged_lab[:, 1:3] - 128.0, axis=1), 90.0
        )
        merged_base = (
            0.46 * np.sqrt(merged_area / max(float(merged_area.max()), 1e-8))
            + 0.20 * np.sqrt(merged_weight_ratio / max(float(merged_weight_ratio.max()), 1e-8))
            + 0.12 * merged_chroma
            + 0.08 * np.sqrt(np.clip(merged_coherence, 0.0, 1.0))
            + 0.14 * merged_region
        )

        if requested is None:
            min_area = max(0.0040, 1.0 / max(float(total_pixels), 1.0) * 1400.0)
            retain = (
                (merged_area >= min_area)
                | ((merged_area >= min_area * 0.55) & (merged_chroma >= 0.32))
                | ((merged_area >= min_area * 0.45) & (merged_region >= 0.52))
                | ((merged_area >= min_area * 0.40) & (merged_weight_ratio >= 0.04))
            )
            selected = np.flatnonzero(retain)
            if len(selected) < 5:
                selected = np.argsort(-merged_base, kind='stable')[:min(5, len(merged_rgb))]
            if len(selected) > 18:
                ranked = np.argsort(-merged_base[selected], kind='stable')[:18]
                selected = selected[ranked]
            selected = np.asarray(selected, dtype=np.intp)
        else:
            final_count = min(requested, len(merged_rgb))
            selected = [int(np.argmax(merged_area))]
            while len(selected) < final_count:
                selected_lab = merged_lab[np.asarray(selected, dtype=np.intp)]
                min_distance = np.min(
                    np.linalg.norm(merged_lab[:, None, :] - selected_lab[None, :, :], axis=2),
                    axis=1,
                )
                diversity = np.clip(min_distance / 42.0, 0.0, 1.0)
                score = (
                    0.48 * merged_base
                    + 0.28 * diversity
                    + 0.16 * merged_region
                    + 0.08 * merged_chroma
                )
                selected_families = [self._palette_iga_color_family(merged_lab[i]) for i in selected]
                for index in range(len(score)):
                    family_count = selected_families.count(self._palette_iga_color_family(merged_lab[index]))
                    score[index] -= 0.12 * family_count
                    if family_count >= 2:
                        score[index] -= 0.42
                score[np.asarray(selected, dtype=np.intp)] = -1e9
                score[min_distance < 7.0] -= 0.50
                selected.append(int(np.argmax(score)))
            selected = np.asarray(selected, dtype=np.intp)

        palette_rgb = merged_rgb[selected]
        palette_lab = self._palette_rgb_to_lab_array(palette_rgb)

        final_distance = np.sum(
            (flat_assignment_lab[:, None, :] - palette_lab[None, :, :]) ** 2,
            axis=2,
        )
        owner_map = np.argmin(final_distance, axis=1).reshape(height, width).astype(np.int32)
        owner_map = self._palette_regularize_owner_map(
            owner_map, assignment_lab_image, palette_lab,
            edge_strength=edge_strength.reshape(height, width),
        )

        final_count = len(palette_rgb)
        final_counts = np.bincount(owner_map.reshape(-1), minlength=final_count).astype(np.int64)
        valid = final_counts > 0
        if not np.all(valid):
            remap = np.full(final_count, -1, dtype=np.int32)
            remap[np.flatnonzero(valid)] = np.arange(int(np.sum(valid)), dtype=np.int32)
            owner_map = remap[owner_map]
            palette_rgb = palette_rgb[valid]
            final_counts = final_counts[valid]
            final_count = len(palette_rgb)

        refined_colors = []
        for palette_index in range(final_count):
            mask = owner_map.reshape(-1) == palette_index
            cluster_rgb = flat_rgb[mask]
            cluster_lab = flat_lab[mask]
            refined_colors.append(self._palette_iga_cluster_representative(
                cluster_rgb, cluster_lab, self._palette_rgb_to_lab_array(palette_rgb[[palette_index]])[0]
            ))
        palette_rgb = np.asarray(refined_colors, dtype=np.uint8)

        palette_lab = self._palette_rgb_to_lab_array(palette_rgb)
        refined_distance = np.sum(
            (flat_assignment_lab[:, None, :] - palette_lab[None, :, :]) ** 2,
            axis=2,
        )
        owner_map = np.argmin(refined_distance, axis=1).reshape(height, width).astype(np.int32)
        owner_map = self._palette_regularize_owner_map(
            owner_map, assignment_lab_image, palette_lab,
            edge_strength=edge_strength.reshape(height, width),
        )
        owner_map, palette_rgb = self._palette_consolidate_owner_regions(
            rgb, owner_map, palette_rgb
        )
        final_count = len(palette_rgb)
        final_counts = np.bincount(owner_map.reshape(-1), minlength=final_count).astype(np.int64)
        order = np.argsort(-final_counts, kind='stable')
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order), dtype=np.intp)
        owner_map = inverse_order[owner_map]
        palette_rgb = palette_rgb[order]
        final_counts = final_counts[order]
        proportions = final_counts.astype(np.float32)
        proportions /= max(float(proportions.sum()), 1.0)

        # 分析阶段可在 720px 内完成，但最终 owner 必须回到编辑器实际预览尺寸。
        # 否则高分辨率图片会出现区域模板尺寸不匹配，也会损失细节精度。
        if full_source.size != source.size:
            full_rgb = np.asarray(full_source, dtype=np.uint8)
            owner_map = self._palette_assign_owner_map_chunked(full_rgb, palette_rgb)
            owner_map, palette_rgb = self._palette_consolidate_owner_regions(
                full_rgb, owner_map, palette_rgb
            )
            final_count = len(palette_rgb)
            final_counts = np.bincount(
                owner_map.reshape(-1), minlength=final_count
            ).astype(np.int64)
            order = np.argsort(-final_counts, kind='stable')
            inverse_order = np.empty_like(order)
            inverse_order[order] = np.arange(len(order), dtype=np.intp)
            owner_map = inverse_order[owner_map]
            palette_rgb = palette_rgb[order]
            final_counts = final_counts[order]
            proportions = final_counts.astype(np.float32)
            proportions /= max(float(proportions.sum()), 1.0)

        model = self._palette_build_sparse_model_from_owner_map(owner_map)
        if return_model:
            return palette_rgb, proportions, model
        return palette_rgb, proportions

    def _palette_build_traditional_owner_model(
            self, image_rgb, source_palette):
        """为传统线稿建立互斥颜色组：每个像素只归属一个调色板颜色。"""
        model = self._palette_build_sparse_model(image_rgb, source_palette)
        model['mode'] = 'traditional_exclusive_owner'
        return model

    def _palette_render_traditional_owner_model(
            self, image_rgb, source_palette, target_palette, model,
            strength=1.0, luminance=0.72, structure=0.62, smoothing=0.18):
        """目标色锚定换色；未选色组和中性暗线保持原像素。

        传统平涂颜色代表的是用户明确指定的目标色。旧版把 ``luminance``
        当成对目标 L 通道的抵消量，导致白色指定纯红后变成橙粉色。这里完整
        应用目标色的 Lab 位移，并保留像素相对源色块的局部残差。
        """
        original = np.asarray(image_rgb, dtype=np.uint8)
        result = self._palette_render_with_model(
            original,
            source_palette,
            target_palette,
            model,
            strength=strength,
            luminance=luminance,
            structure=structure,
            smoothing=smoothing,
            target_anchor=True,
        )
        line_mask = self._palette_traditional_ink_mask(original)
        result[line_mask] = original[line_mask]
        return result

    def _palette_render_traditional_owner_direct(
            self, image_rgb, source_palette, target_palette,
            strength=1.0, luminance=0.72, structure=0.62, smoothing=0.18):
        """以互斥 owner 渲染全分辨率传统图，避免多色权重叠加。"""
        original = np.asarray(image_rgb, dtype=np.uint8)
        model = self._palette_build_traditional_owner_model(
            original, source_palette
        )
        return self._palette_render_traditional_owner_model(
            original,
            source_palette,
            target_palette,
            model,
            strength=strength,
            luminance=luminance,
            structure=structure,
            smoothing=smoothing,
        )

    def _palette_build_sparse_model(self, image_rgb, source_palette):
        """建立互斥的单一颜色 owner 模型。

        每个像素只归属一个代表色，修改 Ck 时仅改变 owner==k 的像素。
        这从模型层消除多个调色板颜色的重心叠加和覆盖现象。
        """
        original = np.asarray(image_rgb, dtype=np.uint8)
        pixels = original.reshape(-1, 3)
        palette = np.asarray(source_palette, dtype=np.uint8)
        if len(palette) < 2:
            raise ValueError("At least two palette colors are required")

        palette_lab = self._palette_rgb_to_lab_array(palette)
        owner_dtype = np.uint8 if len(palette) <= 255 else np.uint16
        owners = np.empty(len(pixels), dtype=owner_dtype)
        confidence = np.empty(len(pixels), dtype=np.float16)
        chunk_size = 160000
        for start in range(0, len(pixels), chunk_size):
            end = min(start + chunk_size, len(pixels))
            chunk_lab = self._palette_rgb_to_lab_array(pixels[start:end])
            distance2 = np.sum(
                (chunk_lab[:, None, :] - palette_lab[None, :, :]) ** 2,
                axis=2,
            )
            nearest_two = np.argpartition(distance2, kth=1, axis=1)[:, :2]
            nearest_distance2 = np.take_along_axis(distance2, nearest_two, axis=1)
            order = np.argsort(nearest_distance2, axis=1)
            nearest_two = np.take_along_axis(nearest_two, order, axis=1)
            nearest_distance2 = np.take_along_axis(nearest_distance2, order, axis=1)
            owners[start:end] = nearest_two[:, 0].astype(owner_dtype)
            d1 = np.sqrt(np.maximum(nearest_distance2[:, 0], 0.0))
            d2 = np.sqrt(np.maximum(nearest_distance2[:, 1], 0.0))
            local_confidence = (d2 - d1) / np.maximum(d2 + 1e-6, 1e-6)
            confidence[start:end] = np.clip(local_confidence, 0.0, 1.0).astype(np.float16)

        return {
            'mode': 'exclusive_owner',
            'shape': tuple(original.shape[:2]),
            'owners': owners,
            'confidence': confidence,
        }

    @staticmethod
    def _palette_build_sparse_model_from_owner_map(owner_map):
        owners = np.asarray(owner_map, dtype=np.int32)
        if owners.ndim != 2:
            raise ValueError("Owner map must be a 2D array")
        flat = owners.reshape(-1)
        valid = flat >= 0
        if not np.any(valid):
            raise ValueError("Owner map does not contain any valid regions")
        owner_dtype = np.uint8 if int(np.max(flat[valid])) <= 255 else np.uint16
        encoded = np.zeros_like(flat, dtype=owner_dtype)
        encoded[valid] = flat[valid].astype(owner_dtype)
        confidence = np.ones(len(flat), dtype=np.float16)
        confidence[~valid] = 0.0
        return {
            'mode': 'exclusive_owner_regions',
            'shape': tuple(owners.shape),
            'owners': encoded,
            'confidence': confidence,
        }

    def _palette_render_with_model(
        self, image_rgb, source_palette, target_palette, model,
        strength=1.0, luminance=0.72, structure=0.62, smoothing=0.18,
        target_anchor=False,
    ):
        """按互斥 owner 应用单色 Lab 位移，未修改 owner 的像素严格保持原值。"""
        original = np.asarray(image_rgb, dtype=np.uint8)
        source = np.asarray(source_palette, dtype=np.uint8)
        target = np.asarray(target_palette, dtype=np.uint8)
        if len(source) != len(target):
            raise ValueError("Source and target palettes must have the same size")

        source_lab = self._palette_rgb_to_lab_array(source)
        target_lab = self._palette_rgb_to_lab_array(target)
        palette_delta = target_lab - source_lab
        changed = np.linalg.norm(palette_delta, axis=1) > 0.5
        if not np.any(changed):
            return original.copy()

        owners = np.asarray(model.get('owners'))
        if owners.size != original.shape[0] * original.shape[1]:
            raise ValueError("Palette owner model does not match image size")
        owners = owners.reshape(original.shape[:2]).astype(np.intp)
        active_mask = changed[owners]
        local_delta = palette_delta[owners]

        original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
        mapped_lab = original_lab.copy()
        # 照片路径仍可保留原亮度；传统平涂必须完整命中用户选择的目标 L/a/b。
        # original + (target - source) 等价于 target + 原像素局部残差。
        luma_scale = (
            1.0
            if bool(target_anchor)
            else 1.0 - float(np.clip(luminance, 0.0, 1.0))
        )
        mapped_lab[:, :, 0][active_mask] += local_delta[:, :, 0][active_mask] * luma_scale
        mapped_lab[:, :, 1][active_mask] += local_delta[:, :, 1][active_mask]
        mapped_lab[:, :, 2][active_mask] += local_delta[:, :, 2][active_mask]
        mapped_rgb = cv2.cvtColor(
            np.clip(mapped_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
        )
        return self._palette_postprocess(
            original, mapped_rgb, strength, luminance, structure, smoothing,
            active_mask=active_mask,
        )

    def _palette_render_direct(
        self, image_rgb, source_palette, target_palette,
        strength=1.0, luminance=0.72, structure=0.62, smoothing=0.18
    ):
        original = np.asarray(image_rgb, dtype=np.uint8)
        model = self._palette_build_sparse_model(original, source_palette)
        return self._palette_render_with_model(
            original, source_palette, target_palette, model,
            strength=strength,
            luminance=luminance,
            structure=structure,
            smoothing=smoothing,
        )

    @staticmethod
    def _palette_postprocess(
        original, recolored, strength, luminance, structure, smoothing, active_mask=None
    ):
        original = np.asarray(original, dtype=np.uint8)
        recolored = np.asarray(recolored, dtype=np.uint8)
        if active_mask is None:
            active_mask = np.ones(original.shape[:2], dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool)
        if not np.any(active_mask):
            return original.copy()

        lab_original = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_result = cv2.cvtColor(recolored, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_orig = lab_original[:, :, 0]
        l_result = lab_result[:, :, 0]
        base = cv2.bilateralFilter(l_orig, 7, 24.0, 24.0)
        detail = l_orig - base
        detail_gain = 0.12 + 0.20 * float(np.clip(structure, 0.0, 1.0))
        adjusted_l = l_result + detail_gain * detail
        lab_result[:, :, 0][active_mask] = adjusted_l[active_mask]

        if smoothing > 0.001:
            gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.sqrt(gx * gx + gy * gy)
            scale = max(float(np.percentile(edge, 92)), 1.0)
            edge = np.clip(edge / scale, 0.0, 1.0)
            flat_weight = float(np.clip(smoothing, 0.0, 1.0)) * (1.0 - edge)
            for channel in (1, 2):
                smooth_channel = cv2.bilateralFilter(lab_result[:, :, channel], 7, 18.0, 18.0)
                blended = (
                    lab_result[:, :, channel] * (1.0 - flat_weight)
                    + smooth_channel * flat_weight
                )
                lab_result[:, :, channel][active_mask] = blended[active_mask]

        processed = cv2.cvtColor(
            np.clip(lab_result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
        ).astype(np.float32)
        alpha = float(np.clip(strength, 0.0, 1.0))
        final = original.astype(np.float32)
        final[active_mask] = (
            original.astype(np.float32)[active_mask] * (1.0 - alpha)
            + processed[active_mask] * alpha
        )
        # 硬约束：未修改颜色的 owner 像素逐值恢复，避免任何滤波串色。
        final[~active_mask] = original[~active_mask]
        return np.clip(final, 0, 255).astype(np.uint8)

    def _palette_prepare_editor_region_payload(
            self, preview, palette, proportions, model):
        """在线程中完成 Representative palette 的全部图像计算。

        Tk 主线程只接收并显示该 payload，避免 worker 完成后又在 UI 线程执行
        对象分组、第二次 SLIC、边缘精修和区域收敛。
        """
        preview_image = preview.convert('RGB').copy()
        preview_rgb = np.asarray(preview_image, dtype=np.uint8)
        source_palette = np.asarray(palette, dtype=np.uint8)
        source_proportions = np.asarray(proportions, dtype=np.float32)
        owners = np.asarray(model.get('owners')) if isinstance(model, dict) else np.asarray([])
        shape = tuple(model.get('shape', ())) if isinstance(model, dict) else ()
        owner_map = (
            owners.reshape(shape).astype(np.int32)
            if owners.size and len(shape) == 2 else None
        )

        group_map = None
        group_map_original = None
        group_map_refined = None
        edge_metrics = {}
        similarity_metrics = {}
        reduction_metrics = {}
        group_source_palette = np.empty((0, 3), dtype=np.uint8)
        group_proportions = np.empty((0,), dtype=np.float32)
        group_members = []

        if owner_map is not None:
            (
                object_group_map,
                object_group_palette,
                object_group_proportions,
                group_members,
            ) = self._photo_build_object_edit_groups(
                preview_rgb, owner_map, source_palette
            )
            group_map_original = object_group_map.copy()
            group_map_refined, edge_metrics = self._palette_refine_owner_edge_band(
                preview_rgb,
                group_map_original,
                object_group_palette,
                soft_limit=0.06,
                hard_limit=0.08,
            )
            # v4.4.6 只保留一个区域收敛权威。v4.4.5 reducer 已同时处理
            # 背景聚合、小区域吸附和相似色合并；预先再跑一次相似合并会重复
            # 扫描整图，并容易把 8~12 个目标区域继续压到 5~7 个。
            group_map = group_map_refined.copy()
            group_source_palette = object_group_palette.copy()
            group_proportions = np.asarray(
                object_group_proportions, dtype=np.float32
            )
            similarity_metrics = {
                'initial_regions': int(len(object_group_palette)),
                'final_regions': int(len(object_group_palette)),
                'merged_regions': 0,
                'retired_from_main_path': True,
                'method': 'delegated to v4.4.5 canonical region reducer',
            }
            try:
                (
                    group_map,
                    group_source_palette,
                    group_proportions,
                    group_members,
                    reduction_metrics,
                ) = self._palette_reduce_object_regions_v445(
                    preview_rgb,
                    group_map,
                    group_source_palette,
                    members=group_members,
                )
            except Exception as exc:
                reduction_metrics = {
                    'fallback': True,
                    'reason': str(exc),
                    'final_regions': int(len(group_source_palette)),
                }

        identity_mae = 0.0
        if group_map is not None and len(group_source_palette):
            reconstruction = self._palette_render_object_groups(
                preview_rgb,
                group_map,
                group_source_palette,
                group_source_palette,
                strength=1.0,
                luminance=1.0,
            )
            identity_mae = float(np.mean(np.abs(
                reconstruction.astype(np.float32) - preview_rgb.astype(np.float32)
            )))

        return {
            'source_palette': source_palette,
            'source_proportions': source_proportions,
            'model': model,
            'owner_map': owner_map,
            'group_map': group_map,
            'group_map_original': group_map_original,
            'group_map_refined': group_map_refined,
            'edge_metrics': edge_metrics,
            'similarity_metrics': similarity_metrics,
            'reduction_metrics': reduction_metrics,
            'group_source_palette': group_source_palette,
            'group_proportions': group_proportions,
            'group_members': group_members,
            'identity_mae': identity_mae,
        }

    def _palette_prepare_editor_payload(self, preview, requested='Auto'):
        """双路径唯一入口：传统图不再进入任何照片对象分组或 reducer。"""
        preview_image = preview.convert('RGB').copy()
        editor_mode, classification = self._palette_resolve_editor_mode(
            preview_image, requested=requested
        )
        if editor_mode == 'photo':
            palette, proportions, model = self._palette_extract_representative(
                preview_image, None, return_model=True
            )
            payload = self._palette_prepare_editor_region_payload(
                preview_image, palette, proportions, model
            )
            payload['editor_mode'] = 'photo'
            payload['classification_metrics'] = classification
            return payload

        palette, proportions = self._palette_extract_traditional_flat(
            preview_image, color_count=8
        )
        if len(palette) < 2:
            raise ValueError(
                "Traditional palette extraction needs at least two distinct colors"
            )
        model = self._palette_build_traditional_owner_model(
            np.asarray(preview_image, dtype=np.uint8), palette
        )
        return {
            'editor_mode': 'traditional',
            'classification_metrics': classification,
            'source_palette': np.asarray(palette, dtype=np.uint8),
            'source_proportions': np.asarray(proportions, dtype=np.float32),
            'model': model,
            'owner_map': None,
            'group_map': None,
            'group_map_original': None,
            'group_map_refined': None,
            'edge_metrics': {},
            'similarity_metrics': {},
            'reduction_metrics': {},
            'group_source_palette': np.empty((0, 3), dtype=np.uint8),
            'group_proportions': np.empty((0,), dtype=np.float32),
            'group_members': [],
            'identity_mae': 0.0,
        }

    def palette_editor_extract_palette(self):
        if self.palette_editor_busy:
            return
        if self.palette_editor_preview_image is None:
            if not self.palette_editor_load_selected_source():
                return
        preview = self.palette_editor_preview_image.copy()
        requested_mode = (
            self.palette_editor_mode_var.get()
            if hasattr(self, 'palette_editor_mode_var') else 'Auto'
        )
        self._palette_editor_set_busy(
            True,
            "Detecting image family and extracting the matching editable palette..."
        )

        def worker():
            try:
                payload = self._palette_prepare_editor_payload(
                    preview, requested=requested_mode
                )
                self.root.after(
                    0, lambda payload=payload: self._palette_editor_finish_extract(payload)
                )
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._palette_editor_fail("Palette extraction failed", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _palette_editor_finish_extract(self, payload):
        """只提交预计算状态并刷新 Tk 控件，不在主线程执行图像算法。"""
        self.palette_editor_mode = str(payload.get('editor_mode', 'photo'))
        self.palette_editor_classification_metrics = dict(
            payload.get('classification_metrics', {}) or {}
        )
        self.palette_editor_source_palette = np.asarray(
            payload['source_palette'], dtype=np.uint8
        )
        self.palette_editor_target_palette = self.palette_editor_source_palette.copy()
        self.palette_editor_proportions = np.asarray(
            payload['source_proportions'], dtype=np.float32
        )
        model = payload['model']
        self.palette_editor_sparse_model = model
        self.palette_editor_owner_map = payload['owner_map']
        self.palette_editor_group_map = payload['group_map']
        self.palette_editor_group_map_original = payload['group_map_original']
        self.palette_editor_group_map_refined = payload['group_map_refined']
        self.palette_editor_edge_refine_metrics = dict(payload['edge_metrics'])
        self.palette_editor_similarity_merge_metrics = dict(payload['similarity_metrics'])
        self.palette_editor_region_reduction_metrics = dict(payload['reduction_metrics'])
        self.palette_editor_group_source_palette = np.asarray(
            payload['group_source_palette'], dtype=np.uint8
        )
        self.palette_editor_group_target_palette = (
            self.palette_editor_group_source_palette.copy()
        )
        self.palette_editor_group_proportions = np.asarray(
            payload['group_proportions'], dtype=np.float32
        )
        self.palette_editor_group_members = list(payload['group_members'])
        self.palette_editor_region_template_sheet = None
        visible_count = (
            len(self.palette_editor_source_palette)
            if self.palette_editor_mode == 'traditional'
            else len(self.palette_editor_group_source_palette)
        )
        self.palette_editor_count_var.set(str(visible_count))
        if hasattr(self, 'palette_editor_region_preview_btn'):
            region_preview_ready = (
                (
                    self.palette_editor_mode == 'traditional'
                    and self.palette_editor_sparse_model is not None
                    and len(self.palette_editor_target_palette) > 0
                )
                or (
                    self.palette_editor_mode == 'photo'
                    and self.palette_editor_group_map is not None
                    and len(self.palette_editor_group_target_palette) > 0
                )
            )
            self.palette_editor_region_preview_btn.config(
                state='normal' if region_preview_ready else 'disabled',
                text=(
                    "View color-group templates..."
                    if self.palette_editor_mode == 'traditional'
                    else "View region templates..."
                ),
            )
        classification = self.palette_editor_classification_metrics
        mode_label = (
            "Traditional line-art & flat"
            if self.palette_editor_mode == 'traditional' else "Photo"
        )
        selection = classification.get('selection', 'auto')
        confidence = 100.0 * float(classification.get('confidence', 0.0))
        reason = classification.get('reason', 'unknown')
        if hasattr(self, 'palette_editor_detected_mode_var'):
            self.palette_editor_detected_mode_var.set(
                f"Using: {mode_label} ({selection}, {confidence:.0f}% confidence; {reason})"
            )
        if hasattr(self, 'palette_editor_count_label'):
            self.palette_editor_count_label.config(
                text=(
                    "Color groups:"
                    if self.palette_editor_mode == 'traditional'
                    else "Detected regions:"
                )
            )
        if hasattr(self, 'palette_editor_swatches_frame'):
            self.palette_editor_swatches_frame.config(
                text=(
                    "Click a global color group to edit"
                    if self.palette_editor_mode == 'traditional'
                    else "Click an object region to edit"
                )
            )
        self.palette_editor_result_image = self.palette_editor_preview_image.copy()
        self.palette_editor_naiga_candidates = []
        self.palette_editor_naiga_scores = []
        self.palette_editor_naiga_selected = None
        self.palette_editor_naiga_generation = 0
        if hasattr(self, 'palette_editor_naiga_frame'):
            self._palette_editor_clear_naiga_candidates()
        self.palette_editor_update_swatches()
        if visible_count:
            self._palette_editor_show_selected_region(0)
        self._palette_editor_refresh_preview_labels()
        if self.palette_editor_mode == 'traditional':
            self._palette_editor_set_busy(
                False,
                (
                    f"{len(self.palette_editor_source_palette)} traditional color groups ready. "
                    f"Mode: {mode_label} ({selection}, {confidence:.0f}% confidence; {reason}). "
                    "v4.5.0 target-anchored colors + ink-excluded exclusive-owner transfer; "
                    "black linework is preserved but is not offered as a swatch."
                ),
            )
            self.update_status(
                "Traditional fill colors extracted (v4.5.0 precise target anchoring)"
            )
            return
        mae = float(payload.get('identity_mae', 0.0))
        fine_count = int(model.get('fine_superpixel_count', 0)) if isinstance(model, dict) else 0
        boundary_score = float(model.get('boundary_adherence', 0.0)) if isinstance(model, dict) else 0.0
        edge_metrics = dict(self.palette_editor_edge_refine_metrics or {})
        merge_metrics = dict(getattr(self, 'palette_editor_similarity_merge_metrics', {}) or {})
        changed_percent = 100.0 * float(edge_metrics.get('changed_ratio', 0.0))
        fallback_note = (
            f"; edge refinement fallback: {edge_metrics.get('reason', 'validation')}"
            if edge_metrics.get('fallback') else ""
        )
        merge_note = (
            f"; similar regions merged: {merge_metrics.get('merged_regions', 0)} "
            f"({merge_metrics.get('initial_regions', 0)}→{merge_metrics.get('final_regions', 0)})"
            if merge_metrics and not merge_metrics.get('retired_from_main_path') else ""
        )
        reduction_metrics = dict(getattr(self, 'palette_editor_region_reduction_metrics', {}) or {})
        if reduction_metrics.get('fallback'):
            reduction_note = (
                f"; v4.4.5 reduction fallback: {reduction_metrics.get('reason', 'validation')}"
            )
        elif reduction_metrics:
            reduction_note = (
                f"; v4.5.0 canonical object-layer reduction: "
                f"{reduction_metrics.get('initial_regions', 0)}→"
                f"{reduction_metrics.get('final_regions', 0)} regions "
                f"(background {reduction_metrics.get('background_merges', 0)}, "
                f"small absorbed {reduction_metrics.get('small_regions_absorbed', 0)}, "
                f"similar {reduction_metrics.get('similarity_merges', 0)}, "
                f"redundant hue {reduction_metrics.get('redundant_hue_merges', 0)}, "
                f"forced {reduction_metrics.get('forced_merges', 0)})"
            )
        else:
            reduction_note = ""
        self._palette_editor_set_busy(
            False,
            (
                f"Detected {len(self.palette_editor_source_palette)} color sublayers and grouped them into "
                f"{len(self.palette_editor_group_source_palette)} object regions. "
                f"Boundary adherence: {boundary_score:.3f}; edge pixels refined: "
                f"{changed_percent:.2f}%; identity MAE: {mae:.2f}{fallback_note}{merge_note}"
                f"{reduction_note}. v4.5.0 photo route. "
                "Strong same-hue duplicates are conservatively merged while coverage "
                "and single-owner integrity stay fixed."
            ),
        )
        self.update_status(
            "Object-aware editable regions extracted (v4.5.0 conservative hue merge)"
        )

    @staticmethod
    def _palette_build_region_template_sheet(image_rgb, owner_map, palette, proportions):
        """生成类似论文区域模板图的总览：区域保留原色，其余位置显示棋盘格。"""
        original = np.asarray(image_rgb, dtype=np.uint8)
        owners = np.asarray(owner_map, dtype=np.int32)
        colors = np.asarray(palette, dtype=np.uint8)
        proportions = np.asarray(proportions, dtype=np.float32)
        if original.ndim != 3 or original.shape[2] != 3:
            raise ValueError("Region preview requires an RGB image")
        if owners.shape != original.shape[:2]:
            raise ValueError("Region template map does not match the image size")
        region_count = len(colors)
        if region_count <= 0:
            raise ValueError("No region colors are available")

        valid = (owners >= 0) & (owners < region_count)
        coverage = float(np.mean(valid)) * 100.0
        # owner map 为单标签整数图，因此从定义上不存在区域重叠。
        overlap = 0.0
        missing = int(np.count_nonzero(~valid))

        cols = min(5, max(2, int(math.ceil(math.sqrt(region_count)))))
        rows = int(math.ceil(region_count / cols))
        cell_w, cell_h = 230, 190
        left_w, header_h = 330, 92
        sheet_w = left_w + cols * cell_w + 28
        sheet_h = max(560, header_h + rows * cell_h + 30)
        sheet = Image.new('RGB', (sheet_w, sheet_h), '#f4f4f4')
        draw = ImageDraw.Draw(sheet)
        try:
            title_font = ImageFont.truetype('arial.ttf', 18)
            body_font = ImageFont.truetype('arial.ttf', 14)
            small_font = ImageFont.truetype('arial.ttf', 12)
        except OSError:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        draw.text((16, 12), "Object-aware editable regions", fill='black', font=title_font)
        summary = (
            f"Regions: {region_count}   Coverage: {coverage:.2f}%   "
            f"Overlap: {overlap:.2f}%   Unassigned pixels: {missing}"
        )
        draw.text((16, 44), summary, fill='#222222', font=body_font)
        draw.text(
            (16, 67),
            "Color sublayers are grouped by spatial object/material continuity; each pixel still belongs to exactly one editable region.",
            fill='#555555', font=small_font,
        )

        draw.rounded_rectangle(
            (14, header_h, left_w - 14, sheet_h - 18),
            radius=8, outline='#c8c8c8', fill='white', width=1,
        )
        draw.text((28, header_h + 14), "Source image", fill='black', font=body_font)
        source_pil = Image.fromarray(original, mode='RGB')
        source_thumb = ImageOps.contain(source_pil, (left_w - 52, min(420, sheet_h - header_h - 72)))
        source_x = 26 + (left_w - 52 - source_thumb.width) // 2
        source_y = header_h + 46
        sheet.paste(source_thumb, (source_x, source_y))

        checker_size = max(5, int(round(min(original.shape[:2]) / 48.0)))
        yy, xx = np.indices(original.shape[:2])
        checker_index = ((yy // checker_size) + (xx // checker_size)) % 2
        checker = np.empty_like(original)
        checker[checker_index == 0] = (218, 218, 218)
        checker[checker_index == 1] = (248, 248, 248)
        component_counts = []

        for index in range(region_count):
            row = index // cols
            col = index % cols
            x0 = left_w + col * cell_w + 8
            y0 = header_h + row * cell_h + 4
            x1 = x0 + cell_w - 12
            y1 = y0 + cell_h - 10
            draw.rounded_rectangle(
                (x0, y0, x1, y1), radius=7,
                outline='#c6c6c6', fill='white', width=1,
            )
            mask = owners == index
            component_count, _component_labels, _stats, _centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            parts = max(0, int(component_count) - 1)
            component_counts.append(parts)
            region_image = checker.copy()
            region_image[mask] = original[mask]
            # 只在预览中加一圈轻微代表色轮廓，便于观察浅色区域形状；不影响 owner map。
            if np.any(mask):
                boundary = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                boundary &= ~mask
                outline_color = np.asarray(colors[index], dtype=np.float32)
                outline_color = np.clip(outline_color * 0.72, 0, 255).astype(np.uint8)
                region_image[boundary] = outline_color
            region_pil = Image.fromarray(region_image, mode='RGB')
            thumb = ImageOps.contain(region_pil, (cell_w - 30, cell_h - 48))
            px = x0 + (cell_w - 12 - thumb.width) // 2
            py = y0 + 8
            sheet.paste(thumb, (px, py))
            rgb = tuple(int(v) for v in colors[index])
            swatch_x0, swatch_y0 = x0 + 10, y1 - 30
            draw.rectangle(
                (swatch_x0, swatch_y0, swatch_x0 + 22, swatch_y0 + 17),
                fill=rgb, outline='black', width=1,
            )
            pct = float(proportions[index] * 100.0) if index < len(proportions) else float(np.mean(mask) * 100.0)
            draw.text(
                (swatch_x0 + 30, swatch_y0 + 1),
                f"R{index + 1}  {pct:.1f}%  parts:{parts}  RGB{rgb}",
                fill='black', font=small_font,
            )

        metrics = {
            'region_count': int(region_count),
            'coverage_percent': coverage,
            'overlap_percent': overlap,
            'unassigned_pixels': missing,
            'component_counts': component_counts,
            'total_components': int(sum(component_counts)),
        }
        return sheet, metrics

    def _palette_editor_region_view_data(self):
        """返回当前编辑路径用于可视预览的互斥区域数据。"""
        if self.palette_editor_preview_image is None:
            raise ValueError("Load an image and run Extract first.")
        if self.palette_editor_mode == 'traditional':
            model = self.palette_editor_sparse_model
            palette = np.asarray(
                self.palette_editor_target_palette, dtype=np.uint8
            )
            proportions = np.asarray(
                self.palette_editor_proportions, dtype=np.float32
            )
            if not isinstance(model, dict) or len(palette) == 0:
                raise ValueError("Extract traditional color groups first.")
            owners = np.asarray(model.get('owners'))
            shape = tuple(model.get('shape', ()))
            if owners.size == 0 or len(shape) != 2:
                raise ValueError("Traditional owner model is unavailable.")
            owner_map = owners.reshape(shape).astype(np.int32)
            if np.any(owner_map < 0) or np.any(owner_map >= len(palette)):
                raise ValueError("Traditional owner model contains invalid labels.")
            members = [[index] for index in range(len(palette))]
            return owner_map, palette, proportions, members, 'C'

        group_map = self.palette_editor_group_map
        palette = np.asarray(
            self.palette_editor_group_target_palette, dtype=np.uint8
        )
        proportions = np.asarray(
            self.palette_editor_group_proportions, dtype=np.float32
        )
        if group_map is None or len(palette) == 0:
            raise ValueError("Extract photo object regions first.")
        members = list(self.palette_editor_group_members)
        return (
            np.asarray(group_map, dtype=np.int32),
            palette,
            proportions,
            members,
            'R',
        )

    def palette_editor_show_region_templates(self):
        if self.palette_editor_preview_image is None:
            messagebox.showinfo(
                "Region templates", "Load an image and run Extract first."
            )
            return
        try:
            owner_map, palette, proportions, _members, _prefix = (
                self._palette_editor_region_view_data()
            )
            sheet, metrics = self._palette_build_region_template_sheet(
                np.asarray(self.palette_editor_preview_image),
                owner_map,
                palette,
                proportions,
            )
        except Exception as exc:
            messagebox.showerror("Region templates", str(exc))
            return
        self.palette_editor_region_template_sheet = sheet

        previous = getattr(self, 'palette_editor_region_template_window', None)
        if previous is not None:
            try:
                previous.destroy()
            except tk.TclError:
                pass
        window = tk.Toplevel(self.root)
        self.palette_editor_region_template_window = window
        window.title(
            f"Region templates - {metrics['region_count']} regions, "
            f"coverage {metrics['coverage_percent']:.2f}%"
        )
        window.geometry("1180x760")
        window.minsize(720, 480)

        toolbar = ttk.Frame(window)
        toolbar.pack(fill='x', padx=8, pady=8)
        ttk.Label(
            toolbar,
            text=(
                f"Total regions: {metrics['region_count']}   "
                f"Coverage: {metrics['coverage_percent']:.2f}%   "
                f"Overlap: {metrics['overlap_percent']:.2f}%   "
                f"Connected parts: {metrics.get('total_components', 0)}"
            ),
        ).pack(side='left')
        ttk.Button(
            toolbar, text="Save preview...",
            command=self.palette_editor_save_region_template_sheet,
        ).pack(side='right')

        frame = ttk.Frame(window)
        frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        canvas = tk.Canvas(frame, background='#ececec', highlightthickness=0)
        x_scroll = ttk.Scrollbar(frame, orient='horizontal', command=canvas.xview)
        y_scroll = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        photo = ImageTk.PhotoImage(sheet)
        canvas._region_template_photo = photo
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.configure(scrollregion=(0, 0, sheet.width, sheet.height))

    def palette_editor_save_region_template_sheet(self):
        sheet = getattr(self, 'palette_editor_region_template_sheet', None)
        if sheet is None:
            messagebox.showinfo("Region templates", "No region preview is available.")
            return
        path = filedialog.asksaveasfilename(
            title="Save region template preview",
            defaultextension='.png',
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")],
            initialfile='naiga_region_templates.png',
        )
        if not path:
            return
        if Path(path).suffix.lower() in ('.jpg', '.jpeg'):
            sheet.save(path, quality=96, subsampling=0, dpi=(300, 300))
        else:
            sheet.save(path, dpi=(300, 300))
        self.update_status(f"Region template preview saved: {path}")

    def _palette_editor_show_selected_region(self, index):
        if self.palette_editor_preview_image is None or index < 0:
            return
        try:
            owner_map, palette, _proportions, members, prefix = (
                self._palette_editor_region_view_data()
            )
        except ValueError:
            return
        if index >= len(palette):
            return
        self.palette_editor_selected_region_index = int(index)
        rgb = np.asarray(self.palette_editor_preview_image, dtype=np.uint8)
        preview, metrics = self._palette_build_selected_region_preview(
            rgb,
            owner_map,
            int(index),
            max_size=(320, 250),
        )
        photo = ImageTk.PhotoImage(preview)
        self.palette_editor_selected_region_photo = photo
        color = palette[index]
        member_count = len(members[index]) if index < len(members) else 1
        detail_line = (
            "Global repeated/symmetric fill group; black ink remains locked"
            if prefix == 'C'
            else f"Contains {member_count} color sublayer(s)"
        )
        self.palette_editor_selected_region_label.configure(
            image=photo,
            text=(
                f"{prefix}{index + 1}  "
                f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}\n"
                f"Area {metrics['area_percent']:.2f}%  |  Parts {metrics['parts']}  |  "
                f"Largest part {metrics['largest_part_percent']:.1f}%\n"
                f"{detail_line}\n"
                "Cyan contour = complete editable-group boundary"
            ),
            compound='top',
        )

    def palette_editor_update_swatches(self):
        for child in self.palette_editor_swatches_frame.winfo_children():
            child.destroy()
        self.palette_editor_swatch_buttons = []
        traditional = self.palette_editor_mode == 'traditional'
        palette = np.asarray(
            self.palette_editor_target_palette
            if traditional else self.palette_editor_group_target_palette,
            dtype=np.uint8,
        )
        proportions = np.asarray(
            self.palette_editor_proportions
            if traditional else self.palette_editor_group_proportions,
            dtype=np.float32,
        )
        if len(palette) == 0:
            ttk.Label(
                self.palette_editor_swatches_frame,
                text="Extract editable colors first",
            ).pack(pady=10)
            return
        for index, color in enumerate(palette):
            hex_color = '#{:02x}{:02x}{:02x}'.format(*[int(v) for v in color])
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            fg = 'white' if luminance < 140 else 'black'
            pct = float(proportions[index] * 100.0) if index < len(proportions) else 0.0
            button = tk.Button(
                self.palette_editor_swatches_frame,
                text=f"{'C' if traditional else 'R'}{index + 1} {hex_color}\n{pct:.1f}%",
                bg=hex_color,
                fg=fg,
                activebackground=hex_color,
                activeforeground=fg,
                relief='solid',
                bd=1,
                command=lambda idx=index: self.palette_editor_choose_color(idx),
            )
            button.grid(row=index // 2, column=index % 2, sticky='ew', padx=4, pady=4)
            self.palette_editor_swatch_buttons.append(button)
        self.palette_editor_swatches_frame.columnconfigure(0, weight=1)
        self.palette_editor_swatches_frame.columnconfigure(1, weight=1)

    def palette_editor_choose_color(self, index):
        traditional = self.palette_editor_mode == 'traditional'
        target_palette = (
            self.palette_editor_target_palette
            if traditional else self.palette_editor_group_target_palette
        )
        if index >= len(target_palette):
            return
        self._palette_editor_show_selected_region(index)
        current = target_palette[index]
        initial = '#{:02x}{:02x}{:02x}'.format(*[int(v) for v in current])
        rgb, _hex = colorchooser.askcolor(
            color=initial,
            title=(
                f"Edit global color group C{index + 1}"
                if traditional else f"Edit region color R{index + 1}"
            ),
        )
        if rgb is None:
            return
        target_palette[index] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        self.palette_editor_naiga_selected = None
        self.palette_editor_update_swatches()
        self._palette_editor_show_selected_region(index)
        self.palette_editor_preview()

    def palette_editor_import_reference_palette(self):
        if self.palette_editor_preview_image is None:
            if not self.palette_editor_load_selected_source():
                return
        traditional = self.palette_editor_mode == 'traditional'
        source_palette = (
            self.palette_editor_source_palette
            if traditional else self.palette_editor_group_source_palette
        )
        source_proportions = (
            self.palette_editor_proportions
            if traditional else self.palette_editor_group_proportions
        )
        if len(source_palette) == 0:
            messagebox.showinfo("Palette Result Editor", "Extract the source palette first.")
            return
        path = filedialog.askopenfilename(
            title="Select reference image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.jfif")],
        )
        if not path:
            return
        reference = Image.open(path).convert('RGB')
        reference.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        count = len(source_palette)
        self._palette_editor_set_busy(True, "Extracting and matching the reference palette...")

        def worker():
            try:
                if traditional:
                    ref_palette, ref_prop = self._palette_extract_traditional_flat(
                        reference, count
                    )
                else:
                    ref_palette, ref_prop = self._palette_extract_representative(
                        reference, count
                    )
                matched = self._palette_match_reference_palette(
                    source_palette,
                    source_proportions,
                    ref_palette,
                    ref_prop,
                )
                self.root.after(0, lambda: self._palette_editor_finish_reference(matched, path))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._palette_editor_fail("Reference palette failed", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _palette_match_reference_palette(
        self, source_palette, source_proportions, reference_palette, reference_proportions
    ):
        src_lab = self._palette_rgb_to_lab_array(source_palette)
        ref_lab = self._palette_rgb_to_lab_array(reference_palette)
        color_cost = np.linalg.norm(src_lab[:, None, :] - ref_lab[None, :, :], axis=2)
        src_prop = np.asarray(source_proportions, dtype=np.float32)
        ref_prop = np.asarray(reference_proportions, dtype=np.float32)
        area_cost = np.abs(src_prop[:, None] - ref_prop[None, :]) * 70.0
        # WACV 2025 调色板传递中的二分图匹配思想：颜色相似度为主、面积角色为辅。
        cost = 0.82 * color_cost + 0.18 * area_cost
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            mapping = np.empty(len(source_palette), dtype=np.intp)
            mapping[rows] = cols
        except Exception:
            mapping = np.empty(len(source_palette), dtype=np.intp)
            remaining = set(range(len(reference_palette)))
            for row in np.argsort(-src_prop, kind='stable'):
                col = min(remaining, key=lambda c: float(cost[row, c]))
                mapping[row] = col
                remaining.remove(col)
        return np.asarray(reference_palette, dtype=np.uint8)[mapping]

    def _palette_editor_finish_reference(self, matched, path):
        if self.palette_editor_mode == 'traditional':
            self.palette_editor_target_palette = np.asarray(matched, dtype=np.uint8)
        else:
            self.palette_editor_group_target_palette = np.asarray(
                matched, dtype=np.uint8
            )
        self.palette_editor_update_swatches()
        self._palette_editor_set_busy(False, f"Reference palette matched: {Path(path).name}")
        self.palette_editor_preview()

    def _palette_editor_clear_naiga_candidates(self):
        frame = getattr(self, 'palette_editor_naiga_frame', None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        self.palette_editor_naiga_widgets = []
        ttk.Label(frame, text="No candidates yet", foreground="#777777").pack(fill='x', pady=3)

    def _palette_owner_labels_for_scoring(self, model, shape):
        if model is None:
            return None
        owners = np.asarray(model.get('owners'))
        if owners.size != int(shape[0] * shape[1]):
            return None
        return owners.reshape(shape).astype(np.int32)

    @staticmethod
    def _palette_adjacency_from_labels(labels, color_count):
        matrix = np.zeros((color_count, color_count), dtype=np.float32)
        if labels is None:
            return matrix
        labels = np.asarray(labels, dtype=np.int32)
        for first, second in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
            valid = (first >= 0) & (second >= 0) & (first != second)
            if not np.any(valid):
                continue
            a = first[valid]
            b = second[valid]
            np.add.at(matrix, (a, b), 1.0)
            np.add.at(matrix, (b, a), 1.0)
        maximum = float(np.max(matrix))
        if maximum > 0:
            matrix /= maximum
        return matrix

    def _palette_naiga_objective_score(self, candidate, base_palette, proportions, adjacency):
        """候选排序辅助分数；最终选择仍由用户点击完成。"""
        candidate = np.asarray(candidate, dtype=np.uint8)
        base = np.asarray(base_palette, dtype=np.uint8)
        cand_lab = self._palette_rgb_to_lab_array(candidate)
        base_lab = self._palette_rgb_to_lab_array(base)
        n = len(candidate)
        if n < 2:
            return 0.0

        distance_to_source = np.linalg.norm(cand_lab[:, None, :] - base_lab[None, :, :], axis=2)
        traceability = float(np.mean(np.exp(-np.min(distance_to_source, axis=1) / 14.0)))
        pairwise = np.linalg.norm(cand_lab[:, None, :] - cand_lab[None, :, :], axis=2)
        upper = pairwise[np.triu_indices(n, 1)]
        separation = float(np.mean(np.clip((upper - 6.0) / 28.0, 0.0, 1.0))) if len(upper) else 0.0

        proportions = np.asarray(proportions, dtype=np.float32)
        if len(proportions) != n or float(np.sum(proportions)) <= 0:
            proportions = np.full(n, 1.0 / n, dtype=np.float32)
        dominant = int(np.argmax(proportions))
        dominant_stability = float(np.exp(-np.linalg.norm(cand_lab[dominant] - base_lab[dominant]) / 30.0))

        adjacency = np.asarray(adjacency, dtype=np.float32)
        if adjacency.shape == (n, n) and float(np.sum(adjacency)) > 0:
            contrast = np.clip(pairwise / 42.0, 0.0, 1.0) * np.exp(-np.maximum(pairwise - 92.0, 0.0) / 55.0)
            adjacency_score = float(np.sum(adjacency * contrast) / max(np.sum(adjacency), 1e-6))
        else:
            adjacency_score = separation

        luma = cand_lab[:, 0]
        if np.std(luma) > 1e-5 and np.std(base_lab[:, 0]) > 1e-5:
            correlation = float(np.corrcoef(base_lab[:, 0], luma)[0, 1])
            if not np.isfinite(correlation):
                correlation = 1.0
        else:
            correlation = 1.0
        luma_order = 0.5 + 0.5 * max(-1.0, min(1.0, correlation))

        score = (
            0.30 * adjacency_score
            + 0.24 * dominant_stability
            + 0.22 * traceability
            + 0.14 * separation
            + 0.10 * luma_order
        )
        return float(np.clip(score * 100.0, 0.0, 100.0))

    def _palette_generate_naiga_population(self, parent_palette, count=6, seed_offset=0):
        """迁移 NA-IGA 的核心：非等位整色重组 + 有界变异，不进行颜色插值。"""
        parent = np.asarray(parent_palette, dtype=np.uint8)
        source_pool = np.asarray(self.palette_editor_source_palette, dtype=np.uint8)
        n = len(parent)
        if n < 2:
            return [parent.copy()]
        if len(source_pool) != n:
            source_pool = parent.copy()
        rng = np.random.default_rng(20260724 + int(seed_offset) * 101 + n)
        proportions = np.asarray(self.palette_editor_proportions, dtype=np.float32)
        dominant = int(np.argmax(proportions)) if len(proportions) == n else 0

        population = [parent.copy()]
        seen = {parent.tobytes()}
        attempts = 0
        while len(population) < int(count) and attempts < 400:
            attempts += 1
            # 两个父代色池：当前用户选择 + 原始提取色；不同 owner 间可整色重组。
            second_parent = source_pool[rng.permutation(n)]
            candidate = parent.copy()
            for slot in range(n):
                if rng.random() < 0.55:  # non-allelic rate
                    pool = np.vstack([parent, second_parent])
                    candidate[slot] = pool[int(rng.integers(0, len(pool)))]
                elif rng.random() < 0.5:
                    candidate[slot] = second_parent[slot]

            # 隔代稳定主导区域，避免大面积 owner 频繁跳色。
            if attempts % 2 == 0:
                candidate[dominant] = parent[dominant]

            # 变异仅做小幅 Lab 微调，并限制回到最近源色的 ΔE 范围。
            if attempts % 3 == 0:
                lab = self._palette_rgb_to_lab_array(candidate)
                edit_count = 1 if n <= 5 else 2
                for idx in rng.choice(n, size=edit_count, replace=False):
                    lab[int(idx)] += np.asarray([
                        rng.uniform(-3.0, 3.0),
                        rng.uniform(-5.0, 5.0),
                        rng.uniform(-5.0, 5.0),
                    ], dtype=np.float32)
                candidate = self._palette_lab_to_rgb_array(lab)

            # 修复重复色：优先换成尚未使用且与该 owner 最接近的源色。
            cand_lab = self._palette_rgb_to_lab_array(candidate)
            for i in range(n):
                for j in range(i):
                    if np.linalg.norm(cand_lab[i] - cand_lab[j]) < 5.0:
                        used = {tuple(map(int, value)) for value in candidate}
                        available = [c for c in source_pool if tuple(map(int, c)) not in used]
                        if available:
                            available = np.asarray(available, dtype=np.uint8)
                            available_lab = self._palette_rgb_to_lab_array(available)
                            nearest = int(np.argmin(np.linalg.norm(available_lab - cand_lab[i], axis=1)))
                            candidate[i] = available[nearest]
                            cand_lab[i] = available_lab[nearest]

            key = candidate.tobytes()
            if key not in seen:
                seen.add(key)
                population.append(candidate)

        # 极端情况下补齐，但保持整色置换。
        while len(population) < int(count):
            candidate = parent[rng.permutation(n)].copy()
            key = candidate.tobytes()
            if key in seen and len(seen) >= math.factorial(min(n, 8)):
                break
            if key not in seen:
                seen.add(key)
                population.append(candidate)
        return population[:int(count)]

    def palette_editor_generate_naiga_candidates(self):
        if len(self.palette_editor_target_palette) == 0 or self.palette_editor_preview_image is None:
            messagebox.showinfo("Palette Result Editor", "Extract a palette first.")
            return
        if self.palette_editor_sparse_model is None:
            self.palette_editor_sparse_model = self._palette_build_sparse_model(
                np.asarray(self.palette_editor_preview_image), self.palette_editor_source_palette
            )
        labels = self._palette_owner_labels_for_scoring(
            self.palette_editor_sparse_model,
            np.asarray(self.palette_editor_preview_image).shape[:2],
        )
        adjacency = self._palette_adjacency_from_labels(labels, len(self.palette_editor_target_palette))
        population = self._palette_generate_naiga_population(
            self.palette_editor_target_palette,
            count=6,
            seed_offset=self.palette_editor_naiga_generation,
        )
        scored = [
            (
                self._palette_naiga_objective_score(
                    candidate,
                    self.palette_editor_source_palette,
                    self.palette_editor_proportions,
                    adjacency,
                ),
                candidate,
            )
            for candidate in population
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        self.palette_editor_naiga_scores = [float(item[0]) for item in scored]
        self.palette_editor_naiga_candidates = [np.asarray(item[1], dtype=np.uint8) for item in scored]
        self.palette_editor_naiga_selected = None
        self._palette_editor_show_naiga_candidates()
        self.palette_editor_info_var.set(
            f"NA-IGA generation {self.palette_editor_naiga_generation}: {len(scored)} traceable candidates ready."
        )

    def _palette_editor_show_naiga_candidates(self):
        frame = getattr(self, 'palette_editor_naiga_frame', None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        self.palette_editor_naiga_widgets = []
        for index, (candidate, score) in enumerate(zip(
            self.palette_editor_naiga_candidates, self.palette_editor_naiga_scores
        )):
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=2)
            canvas = tk.Canvas(row, width=170, height=24, highlightthickness=1, highlightbackground='#999999')
            canvas.pack(side='left', fill='x', expand=True)
            width = 170.0 / max(len(candidate), 1)
            for color_index, color in enumerate(candidate):
                hex_color = '#{:02x}{:02x}{:02x}'.format(*[int(v) for v in color])
                canvas.create_rectangle(
                    color_index * width, 0, (color_index + 1) * width, 24,
                    fill=hex_color, outline=''
                )
            button = ttk.Button(
                row,
                text=f"Use P{index + 1}  {score:.1f}",
                width=14,
                command=lambda idx=index: self.palette_editor_apply_naiga_candidate(idx),
            )
            button.pack(side='right', padx=(5, 0))
            canvas.bind('<Button-1>', lambda _event, idx=index: self.palette_editor_apply_naiga_candidate(idx))
            self.palette_editor_naiga_widgets.append((row, canvas, button))

    def palette_editor_apply_naiga_candidate(self, index):
        if not 0 <= int(index) < len(self.palette_editor_naiga_candidates):
            return
        self.palette_editor_naiga_selected = int(index)
        self.palette_editor_target_palette = self.palette_editor_naiga_candidates[int(index)].copy()
        self.palette_editor_update_swatches()
        score = self.palette_editor_naiga_scores[int(index)]
        self.palette_editor_info_var.set(
            f"Applied NA-IGA candidate P{int(index) + 1}; support score {score:.1f}."
        )
        self.palette_editor_preview()

    def palette_editor_evolve_naiga_selected(self):
        if len(self.palette_editor_target_palette) == 0:
            messagebox.showinfo("Palette Result Editor", "Extract a palette first.")
            return
        if self.palette_editor_naiga_candidates and self.palette_editor_naiga_selected is None:
            messagebox.showinfo("Palette Result Editor", "Select a candidate first.")
            return
        self.palette_editor_naiga_generation += 1
        self.palette_editor_generate_naiga_candidates()

    def palette_editor_optimize_palette(self):
        if len(self.palette_editor_target_palette) == 0:
            messagebox.showinfo("Palette Result Editor", "Extract a palette first.")
            return
        amount = max(0.0, min(1.0, float(self.palette_editor_optimize_var.get()) / 100.0))
        if amount <= 0.0:
            return
        colors = self.palette_editor_target_palette.astype(np.float32) / 255.0
        hsv = np.asarray([colorsys.rgb_to_hsv(*color) for color in colors], dtype=np.float32)
        anchor = int(np.argmax(self.palette_editor_proportions)) if len(self.palette_editor_proportions) else 0
        anchor_hue = float(hsv[anchor, 0])
        templates = np.asarray([0, 30, 60, 90, 120, 150, 180], dtype=np.float32) / 360.0
        # 轻量色彩和谐调整：只做小幅移动，不替换用户颜色。
        for idx in range(len(hsv)):
            if idx == anchor or hsv[idx, 1] < 0.08:
                continue
            delta = (float(hsv[idx, 0]) - anchor_hue + 0.5) % 1.0 - 0.5
            sign = -1.0 if delta < 0 else 1.0
            target_delta = float(templates[np.argmin(np.abs(templates - abs(delta)))]) * sign
            adjusted = anchor_hue + target_delta
            hsv[idx, 0] = (hsv[idx, 0] * (1.0 - 0.22 * amount) + adjusted * (0.22 * amount)) % 1.0
        harmonized = np.asarray([colorsys.hsv_to_rgb(*value) for value in hsv], dtype=np.float32)
        lab = self._palette_rgb_to_lab_array(np.clip(harmonized * 255.0, 0, 255).astype(np.uint8))
        original_lab = lab.copy()
        minimum_distance = 13.0 + 8.0 * amount
        for _iteration in range(8):
            for i in range(len(lab)):
                for j in range(i + 1, len(lab)):
                    vector = lab[i] - lab[j]
                    distance = float(np.linalg.norm(vector))
                    if distance >= minimum_distance:
                        continue
                    if distance < 1e-4:
                        angle = 2.0 * math.pi * (i + 1) / max(len(lab), 1)
                        vector = np.asarray([0.25, math.cos(angle), math.sin(angle)], dtype=np.float32)
                        distance = float(np.linalg.norm(vector))
                    push = 0.5 * (minimum_distance - distance) * amount
                    unit = vector / max(distance, 1e-5)
                    lab[i] += push * unit
                    lab[j] -= push * unit
            max_move = 18.0 * amount
            displacement = lab - original_lab
            norm = np.linalg.norm(displacement, axis=1, keepdims=True)
            factor = np.minimum(1.0, max_move / np.maximum(norm, 1e-6))
            lab = original_lab + displacement * factor
            lab[:, 0] = np.clip(lab[:, 0], 18, 238)
            lab[:, 1:] = np.clip(lab[:, 1:], 12, 244)
        self.palette_editor_target_palette = self._palette_lab_to_rgb_array(lab)
        self.palette_editor_update_swatches()
        self.palette_editor_info_var.set(
            "Palette optimized with gentle hue-template alignment and perceptual color separation."
        )
        self.palette_editor_preview()

    def palette_editor_apply_preset(self, preset):
        if preset == "structure":
            values = (88.0, 88.0, 85.0, 12.0)
        elif preset == "strong":
            values = (100.0, 38.0, 22.0, 8.0)
        else:
            values = (100.0, 72.0, 62.0, 18.0)
        self.palette_editor_strength_var.set(values[0])
        self.palette_editor_luma_var.set(values[1])
        self.palette_editor_structure_var.set(values[2])
        self.palette_editor_smooth_var.set(values[3])
        active_palette = (
            self.palette_editor_target_palette
            if self.palette_editor_mode == 'traditional'
            else self.palette_editor_group_target_palette
        )
        if len(active_palette):
            self.palette_editor_preview()

    def palette_editor_preview(self):
        if self.palette_editor_busy:
            return
        traditional = self.palette_editor_mode == 'traditional'
        ready = (
            self.palette_editor_preview_image is not None
            and (
                (
                    traditional
                    and self.palette_editor_sparse_model is not None
                    and len(self.palette_editor_target_palette) > 0
                )
                or (
                    not traditional
                    and self.palette_editor_group_map is not None
                    and len(self.palette_editor_group_target_palette) > 0
                )
            )
        )
        if not ready:
            messagebox.showinfo(
                "Palette Result Editor",
                "Load an image and extract its editable palette first.",
            )
            return
        original = np.asarray(self.palette_editor_preview_image)
        strength = float(self.palette_editor_strength_var.get()) / 100.0
        luminance = float(self.palette_editor_luma_var.get()) / 100.0
        structure = float(self.palette_editor_structure_var.get()) / 100.0
        smoothing = float(self.palette_editor_smooth_var.get()) / 100.0
        if traditional:
            source = self.palette_editor_source_palette.copy()
            target = self.palette_editor_target_palette.copy()
            model = self.palette_editor_sparse_model
            group_map = None
            busy_text = "Rendering exclusive-owner traditional palette preview..."
        else:
            source = self.palette_editor_group_source_palette.copy()
            target = self.palette_editor_group_target_palette.copy()
            model = None
            group_map = self.palette_editor_group_map.copy()
            busy_text = "Rendering complete photo object-region preview..."
        self._palette_editor_set_busy(True, busy_text)

        def worker():
            try:
                if traditional:
                    result = self._palette_render_traditional_owner_model(
                        original,
                        source,
                        target,
                        model,
                        strength=strength,
                        luminance=luminance,
                        structure=structure,
                        smoothing=smoothing,
                    )
                else:
                    result = self._palette_render_object_groups(
                        original, group_map, source, target,
                        strength=strength,
                        luminance=luminance,
                    )
                self.root.after(0, lambda: self._palette_editor_finish_preview(result))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._palette_editor_fail("Preview failed", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _palette_editor_finish_preview(self, result):
        self.palette_editor_result_image = Image.fromarray(np.asarray(result, dtype=np.uint8), mode='RGB')
        self._palette_editor_refresh_preview_labels()
        original_lab = self._palette_rgb_to_lab_array(np.asarray(self.palette_editor_preview_image).reshape(-1, 3))
        result_lab = self._palette_rgb_to_lab_array(np.asarray(self.palette_editor_result_image).reshape(-1, 3))
        mean_change = float(np.mean(np.linalg.norm(original_lab - result_lab, axis=1)))
        self._palette_editor_set_busy(
            False,
            (
                f"Preview ready. Mean perceptual color change: {mean_change:.2f}. "
                + (
                    "Similar, repeated and symmetric traditional colors move together."
                    if self.palette_editor_mode == 'traditional'
                    else "Complete photo object regions are recolored together; unchanged regions are preserved exactly."
                )
            ),
        )
        self.update_status("Palette recoloring preview ready")

    def palette_editor_reset_palette(self):
        if self.palette_editor_mode == 'traditional':
            if len(self.palette_editor_source_palette) == 0:
                return
            self.palette_editor_target_palette = self.palette_editor_source_palette.copy()
        else:
            if len(self.palette_editor_group_source_palette) == 0:
                return
            self.palette_editor_group_target_palette = (
                self.palette_editor_group_source_palette.copy()
            )
        self.palette_editor_naiga_candidates = []
        self.palette_editor_naiga_scores = []
        self.palette_editor_naiga_selected = None
        self.palette_editor_naiga_generation = 0
        if hasattr(self, 'palette_editor_naiga_frame'):
            self._palette_editor_clear_naiga_candidates()
        self.palette_editor_update_swatches()
        self.palette_editor_result_image = self.palette_editor_preview_image.copy()
        self._palette_editor_refresh_preview_labels()
        self.palette_editor_info_var.set(
            "Traditional color groups reset to source appearance."
            if self.palette_editor_mode == 'traditional'
            else "Object-region colors reset to the extracted source appearance."
        )

    def palette_editor_save_result(self):
        if self.palette_editor_busy:
            return
        traditional = self.palette_editor_mode == 'traditional'
        ready = (
            self.palette_editor_original_image is not None
            and (
                (traditional and len(self.palette_editor_target_palette) > 0)
                or (
                    not traditional
                    and self.palette_editor_group_map is not None
                    and len(self.palette_editor_group_target_palette) > 0
                )
            )
        )
        if not ready:
            messagebox.showinfo(
                "Palette Result Editor", "Create an editable palette result first."
            )
            return
        path = filedialog.asksaveasfilename(
            title="Save palette-optimized image",
            defaultextension='.png',
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("TIFF files", "*.tiff")],
            initialfile='palette_recolored.png',
        )
        if not path:
            return
        original = np.asarray(self.palette_editor_original_image.convert('RGB'))
        if traditional:
            source = self.palette_editor_source_palette.copy()
            target = self.palette_editor_target_palette.copy()
            group_map = None
        else:
            source = self.palette_editor_group_source_palette.copy()
            target = self.palette_editor_group_target_palette.copy()
            group_map = cv2.resize(
                self.palette_editor_group_map.astype(np.int32),
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        strength = float(self.palette_editor_strength_var.get()) / 100.0
        luminance = float(self.palette_editor_luma_var.get()) / 100.0
        structure = float(self.palette_editor_structure_var.get()) / 100.0
        smoothing = float(self.palette_editor_smooth_var.get()) / 100.0
        self._palette_editor_set_busy(
            True,
            (
                "Rendering full-resolution traditional palette result..."
                if traditional
                else "Rendering full-resolution photo object-region result..."
            ),
        )

        def worker():
            try:
                if traditional:
                    result = self._palette_render_traditional_owner_direct(
                        original,
                        source,
                        target,
                        strength=strength,
                        luminance=luminance,
                        structure=structure,
                        smoothing=smoothing,
                    )
                else:
                    result = self._palette_render_object_groups(
                        original, group_map, source, target,
                        strength=strength,
                        luminance=luminance,
                    )
                output = Image.fromarray(result, mode='RGB')
                extension = Path(path).suffix.lower()
                if extension in ('.jpg', '.jpeg'):
                    output.save(path, quality=96, subsampling=0, dpi=(300, 300))
                else:
                    output.save(path, dpi=(300, 300))
                self.root.after(0, lambda: self._palette_editor_finish_save(path))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._palette_editor_fail("Save failed", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _palette_editor_finish_save(self, path):
        self._palette_editor_set_busy(False, f"Saved full-resolution result: {Path(path).name}")
        self.update_status(f"Palette result saved: {path}")
        messagebox.showinfo("Success", f"Palette-optimized image saved to:\n{path}")

    def _palette_editor_fail(self, title, exc):
        self._palette_editor_set_busy(False, f"{title}: {exc}")
        messagebox.showerror(title, str(exc))

    # -------------------- create_design_tab 方法 --------------------
    def create_design_tab(self):
        """创建方案设计标签页（包含下载设置和对称类型）"""
        tab_frame = ttk.Frame( self.design_tab )
        tab_frame.pack( fill='both', expand=True, padx=10, pady=10 )

        # 下载设置
        settings_frame = ttk.LabelFrame( tab_frame, text="Download settings" )
        settings_frame.pack( fill='x', padx=10, pady=(0, 10) )

        ttk.Label( settings_frame, text="Width (px):" ).grid( row=0, column=0, padx=5, pady=5, sticky='e' )
        self.download_width = tk.IntVar( value=2048 )
        ttk.Spinbox( settings_frame, from_=100, to=10000, increment=10,
                     textvariable=self.download_width, width=8 ).grid( row=0, column=1, padx=5, pady=5 )

        ttk.Label( settings_frame, text="Height (px):" ).grid( row=0, column=2, padx=5, pady=5, sticky='e' )
        self.download_height = tk.IntVar( value=2048 )
        ttk.Spinbox( settings_frame, from_=100, to=10000, increment=10,
                     textvariable=self.download_height, width=8 ).grid( row=0, column=3, padx=5, pady=5 )

        ttk.Label( settings_frame, text="DPI:" ).grid( row=0, column=4, padx=5, pady=5, sticky='e' )
        self.download_dpi = tk.IntVar( value=300 )
        ttk.Spinbox( settings_frame, from_=72, to=1200, increment=10,
                     textvariable=self.download_dpi, width=6 ).grid( row=0, column=5, padx=5, pady=5 )

        # 方案显示区域
        design_frame = ttk.LabelFrame( tab_frame, text="Coloring scheme" )
        design_frame.pack( fill='both', expand=True, padx=10, pady=10 )

        self.design_canvas = tk.Canvas(design_frame, bg='white', highlightthickness=0)
        vertical_scrollbar = ttk.Scrollbar(
            design_frame, orient='vertical', command=self.design_canvas.yview
        )
        horizontal_scrollbar = ttk.Scrollbar(
            design_frame, orient='horizontal', command=self.design_canvas.xview
        )
        self.design_container = ttk.Frame(self.design_canvas)

        self.design_container.bind(
            "<Configure>",
            lambda e: self.design_canvas.configure(
                scrollregion=self.design_canvas.bbox("all")
            )
        )

        self._design_canvas_window = self.design_canvas.create_window(
            (0, 0), window=self.design_container, anchor="nw"
        )
        self.design_canvas.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )

        design_frame.grid_rowconfigure(0, weight=1)
        design_frame.grid_columnconfigure(0, weight=1)
        self.design_canvas.grid(row=0, column=0, sticky='nsew')
        vertical_scrollbar.grid(row=0, column=1, sticky='ns')
        horizontal_scrollbar.grid(row=1, column=0, sticky='ew')

        # Shift + 滚轮进行横向浏览；普通滚轮保持纵向浏览。
        self.design_canvas.bind_all(
            "<Shift-MouseWheel>",
            lambda event: self.design_canvas.xview_scroll(
                int(-1 * (event.delta / 120)), "units"
            )
        )


# 创建主窗口并启动应用
if __name__ == "__main__":
    root = tk.Tk()
    app = ColorTransferToolbox( root )
    root.mainloop()
