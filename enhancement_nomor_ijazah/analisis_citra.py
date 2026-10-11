import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================

INPUT_DIR = Path("Gambar")
OUTPUT_DIR = Path("hasil")

# Daftar 9 citra
gambar_list = [
    ("01", "01_HighQuality_Enhanced.jpg"),
    ("02", "02_LowContrast.jpg"),
    ("03", "03_Blurred.jpg"),
    ("04", "04_HighNoise.jpg"),
    ("05", "05_LowResolution_Upsampled.jpg"),
    ("06", "06_Faded_Underexposed.jpg"),
    ("07", "07_ColorShift_WarmTint.jpg"),
    ("08", "08_JPEGCompression_Artifacts.jpg"),
    ("09", "09_CombinedDegradation.jpg")
]


# ============================================================
# FUNGSI MEMBUAT NAMA FOLDER
# ============================================================

def nama_folder(no, nama_file):
    nama = Path(nama_file).stem
    return f"{no}_{nama}"


# ============================================================
# FUNGSI RESIZE UNTUK MENAMPILKAN CITRA SAAT CROP
# ============================================================

def resize_for_display(image, max_width=1200, max_height=750):
    h, w = image.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# CROP MANUAL
# ============================================================

def crop_manual(image, nomor):
    """
    User memilih sendiri area nomor ijazah.
    """

    tampil, scale = resize_for_display(image)

    print()
    print("=" * 60)
    print(f"CITRA {nomor}")
    print("=" * 60)
    print("Silakan crop AREA NOMOR IJAZAH.")
    print("1. Drag mouse pada area nomor ijazah.")
    print("2. Tekan ENTER setelah selesai.")
    print("3. Tekan ESC jika ingin membatalkan.")
    print()

    roi = cv2.selectROI(
        f"CROP NOMOR IJAZAH - CITRA {nomor}",
        tampil,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("Crop dibatalkan.")
        return None

    # Kembalikan koordinat ke ukuran gambar asli
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # Pastikan tidak keluar batas
    img_h, img_w = image.shape[:2]

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))

    w = min(w, img_w - x)
    h = min(h, img_h - y)

    crop = image[y:y+h, x:x+w]

    return crop


# ============================================================
# BRIGHTNESS ADJUSTMENT
# ============================================================

def brightness_adjustment(gray):
    """
    Menambahkan brightness secukupnya.
    """

    # beta = nilai brightness
    # Nilai positif membuat citra lebih terang
    beta = 15

    hasil = cv2.convertScaleAbs(
        gray,
        alpha=1.0,
        beta=beta
    )

    return hasil


# ============================================================
# CONTRAST STRETCHING
# ============================================================

def contrast_stretching(gray):
    """
    Contrast stretching menggunakan persentil
    agar lebih tahan terhadap noise/outlier.
    """

    low = np.percentile(gray, 2)
    high = np.percentile(gray, 98)

    if high <= low:
        return gray.copy()

    hasil = (gray.astype(np.float32) - low) * 255.0 / (high - low)

    hasil = np.clip(hasil, 0, 255)

    return hasil.astype(np.uint8)


# ============================================================
# HISTOGRAM EQUALIZATION
# ============================================================

def histogram_equalization(gray):
    return cv2.equalizeHist(gray)


# ============================================================
# FUNGSI HISTOGRAM
# ============================================================

def tampilkan_histogram(ax, image, judul):
    """
    Histogram grayscale.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    ax.hist(
        gray.ravel(),
        bins=256,
        range=(0, 256)
    )

    ax.set_title(judul, fontsize=11, fontweight="bold")
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Jumlah Piksel")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)


# ============================================================
# SIMPAN GAMBAR HASIL INDIVIDUAL
# ============================================================

def simpan_citra_individual(
    folder,
    crop_asli,
    brightness,
    contrast,
    equalization
):

    # ========================================================
    # 01 - CITRA ASLI
    # PENTING:
    # Ini tetap WARNA ASLI, bukan grayscale.
    # ========================================================

    cv2.imwrite(
        str(folder / "01_citra_asli.png"),
        crop_asli
    )

    # ========================================================
    # 02 - BRIGHTNESS
    # ========================================================

    cv2.imwrite(
        str(folder / "02_brightness.png"),
        brightness
    )

    # ========================================================
    # 03 - CONTRAST STRETCHING
    # ========================================================

    cv2.imwrite(
        str(folder / "03_contrast_stretching.png"),
        contrast
    )

    # ========================================================
    # 04 - HISTOGRAM EQUALIZATION
    # ========================================================

    cv2.imwrite(
        str(folder / "04_histogram_equalization.png"),
        equalization
    )


# ============================================================
# BUAT GAMBAR PERBANDINGAN CITRA
# ============================================================

def buat_perbandingan_citra(
    folder,
    nomor,
    crop_asli,
    brightness,
    contrast,
    equalization
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8)
    )

    # --------------------------------------------------------
    # SEBELUM ENHANCEMENT
    # TAMPILKAN WARNA ASLI
    # --------------------------------------------------------

    axes[0, 0].imshow(
        cv2.cvtColor(crop_asli, cv2.COLOR_BGR2RGB)
    )

    axes[0, 0].set_title(
        "SEBELUM ENHANCEMENT\n(Citra Asli)",
        fontsize=13,
        fontweight="bold"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    axes[0, 1].imshow(
        brightness,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 1].set_title(
        "BRIGHTNESS ADJUSTMENT",
        fontsize=13,
        fontweight="bold"
    )

    axes[0, 1].axis("off")

    # --------------------------------------------------------
    # CONTRAST STRETCHING
    # --------------------------------------------------------

    axes[1, 0].imshow(
        contrast,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 0].set_title(
        "CONTRAST STRETCHING",
        fontsize=13,
        fontweight="bold"
    )

    axes[1, 0].axis("off")

    # --------------------------------------------------------
    # HISTOGRAM EQUALIZATION
    # --------------------------------------------------------

    axes[1, 1].imshow(
        equalization,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 1].set_title(
        "HISTOGRAM EQUALIZATION",
        fontsize=13,
        fontweight="bold"
    )

    axes[1, 1].axis("off")

    fig.suptitle(
        f"ANALISIS ENHANCEMENT NOMOR IJAZAH - CITRA {nomor}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(
        folder / "HASIL_PERBANDINGAN.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# BUAT PERBANDINGAN HISTOGRAM
# ============================================================

def buat_perbandingan_histogram(
    folder,
    nomor,
    crop_asli,
    gray,
    brightness,
    contrast,
    equalization
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9)
    )

    axes = axes.ravel()

    # --------------------------------------------------------
    # HISTOGRAM SEBELUM ENHANCEMENT
    # --------------------------------------------------------

    tampilkan_histogram(
        axes[0],
        gray,
        "HISTOGRAM SEBELUM ENHANCEMENT"
    )

    # --------------------------------------------------------
    # HISTOGRAM BRIGHTNESS
    # --------------------------------------------------------

    tampilkan_histogram(
        axes[1],
        brightness,
        "HISTOGRAM BRIGHTNESS ADJUSTMENT"
    )

    # --------------------------------------------------------
    # HISTOGRAM CONTRAST STRETCHING
    # --------------------------------------------------------

    tampilkan_histogram(
        axes[2],
        contrast,
        "HISTOGRAM CONTRAST STRETCHING"
    )

    # --------------------------------------------------------
    # HISTOGRAM EQUALIZATION
    # --------------------------------------------------------

    tampilkan_histogram(
        axes[3],
        equalization,
        "HISTOGRAM HISTOGRAM EQUALIZATION"
    )

    fig.suptitle(
        f"PERBANDINGAN HISTOGRAM - NOMOR IJAZAH CITRA {nomor}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(
        folder / "HISTOGRAM_PERBANDINGAN.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# ANALISIS SEDERHANA
# ============================================================

def analisis_nilai(gray, brightness, contrast, equalization):

    hasil = {}

    hasil["Sebelum"] = {
        "min": int(gray.min()),
        "max": int(gray.max()),
        "mean": float(gray.mean()),
        "std": float(gray.std())
    }

    hasil["Brightness"] = {
        "min": int(brightness.min()),
        "max": int(brightness.max()),
        "mean": float(brightness.mean()),
        "std": float(brightness.std())
    }

    hasil["Contrast Stretching"] = {
        "min": int(contrast.min()),
        "max": int(contrast.max()),
        "mean": float(contrast.mean()),
        "std": float(contrast.std())
    }

    hasil["Histogram Equalization"] = {
        "min": int(equalization.min()),
        "max": int(equalization.max()),
        "mean": float(equalization.mean()),
        "std": float(equalization.std())
    }

    return hasil


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" ANALISIS ENHANCEMENT NOMOR IJAZAH")
    print("=" * 70)
    print()
    print("Program akan memproses 9 citra.")
    print("Untuk setiap citra, Anda memilih sendiri area nomor ijazah.")
    print()
    print("Alur:")
    print("Citra asli berwarna")
    print("        ↓")
    print("Crop manual nomor ijazah")
    print("        ↓")
    print("Grayscale")
    print("        ↓")
    print("Brightness / Contrast Stretching / Histogram Equalization")
    print()

    # Buat folder hasil utama
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for nomor, nama_file in gambar_list:

        input_path = INPUT_DIR / nama_file

        print()
        print("=" * 70)
        print(f"MEMPROSES CITRA {nomor}: {nama_file}")
        print("=" * 70)

        # ----------------------------------------------------
        # BACA CITRA
        # ----------------------------------------------------

        image = cv2.imread(
            str(input_path)
        )

        if image is None:
            print()
            print(f"ERROR: File tidak ditemukan:")
            print(input_path)
            print("Citra dilewati.")
            continue

        print(
            f"Ukuran citra asli: "
            f"{image.shape[1]} x {image.shape[0]}"
        )

        # ----------------------------------------------------
        # CROP MANUAL
        # ----------------------------------------------------

        crop_asli = crop_manual(
            image,
            nomor
        )

        if crop_asli is None:
            print(
                f"Citra {nomor} dilewati karena crop dibatalkan."
            )
            continue

        # ----------------------------------------------------
        # BUAT FOLDER OUTPUT
        # ----------------------------------------------------

        folder_name = nama_folder(
            nomor,
            nama_file
        )

        output_folder = OUTPUT_DIR / folder_name

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # SIMPAN CROP ASLI
        # ----------------------------------------------------

        cv2.imwrite(
            str(output_folder / "01_citra_asli.png"),
            crop_asli
        )

        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            crop_asli,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # ENHANCEMENT 1
        # BRIGHTNESS
        # ----------------------------------------------------

        brightness = brightness_adjustment(
            gray
        )

        # ----------------------------------------------------
        # ENHANCEMENT 2
        # CONTRAST STRETCHING
        # ----------------------------------------------------

        contrast = contrast_stretching(
            gray
        )

        # ----------------------------------------------------
        # ENHANCEMENT 3
        # HISTOGRAM EQUALIZATION
        # ----------------------------------------------------

        equalization = histogram_equalization(
            gray
        )

        # ----------------------------------------------------
        # SIMPAN HASIL INDIVIDUAL
        # ----------------------------------------------------

        simpan_citra_individual(
            output_folder,
            crop_asli,
            brightness,
            contrast,
            equalization
        )

        # ----------------------------------------------------
        # BUAT GAMBAR GABUNGAN CITRA
        # ----------------------------------------------------

        buat_perbandingan_citra(
            output_folder,
            nomor,
            crop_asli,
            brightness,
            contrast,
            equalization
        )

        # ----------------------------------------------------
        # BUAT GAMBAR GABUNGAN HISTOGRAM
        # ----------------------------------------------------

        buat_perbandingan_histogram(
            output_folder,
            nomor,
            crop_asli,
            gray,
            brightness,
            contrast,
            equalization
        )

        # ----------------------------------------------------
        # ANALISIS NILAI
        # ----------------------------------------------------

        data = analisis_nilai(
            gray,
            brightness,
            contrast,
            equalization
        )

        print()
        print("HASIL ANALISIS CITRA", nomor)
        print("-" * 70)

        for metode, nilai in data.items():

            print(
                f"{metode:<25} "
                f"Min={nilai['min']:3d} | "
                f"Max={nilai['max']:3d} | "
                f"Mean={nilai['mean']:.2f} | "
                f"Std={nilai['std']:.2f}"
            )

        print()
        print("Hasil disimpan di:")
        print(output_folder)

    # ========================================================
    # SELESAI
    # ========================================================

    print()
    print("=" * 70)
    print("SEMUA CITRA SELESAI DIPROSES")
    print("=" * 70)
    print()
    print("Folder hasil:")
    print(OUTPUT_DIR.resolve())
    print()
    print("Setiap citra memiliki:")
    print("1. 01_citra_asli.png")
    print("2. 02_brightness.png")
    print("3. 03_contrast_stretching.png")
    print("4. 04_histogram_equalization.png")
    print("5. HASIL_PERBANDINGAN.png")
    print("6. HISTOGRAM_PERBANDINGAN.png")
    print()


# ============================================================
# JALANKAN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()