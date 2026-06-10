import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('doc1.png')

if img is None:
    print("Image not found. Check the file path.")
    exit()

img = cv2.resize(img, (256, 256))

b, g, r = cv2.split(img)

def dct2(channel):
    return cv2.dct(np.float32(channel))

def idct2(channel):
    return cv2.idct(channel)

def compress_channel(dct_channel, compression_ratio):
    rows, cols = dct_channel.shape

    crow = int(rows * compression_ratio)
    ccol = int(cols * compression_ratio)

    dct_channel[crow:, :] = 0
    dct_channel[:, ccol:] = 0

    return dct_channel

compression_ratio = 0.1

dct_b = dct2(b)
dct_g = dct2(g)
dct_r = dct2(r)

dct_b_compressed = compress_channel(np.copy(dct_b), compression_ratio)
dct_g_compressed = compress_channel(np.copy(dct_g), compression_ratio)
dct_r_compressed = compress_channel(np.copy(dct_r), compression_ratio)

b_reconstructed = idct2(dct_b_compressed)
g_reconstructed = idct2(dct_g_compressed)
r_reconstructed = idct2(dct_r_compressed)

b_reconstructed = np.clip(b_reconstructed, 0, 255).astype(np.uint8)
g_reconstructed = np.clip(g_reconstructed, 0, 255).astype(np.uint8)
r_reconstructed = np.clip(r_reconstructed, 0, 255).astype(np.uint8)

img_reconstructed = cv2.merge(
    (b_reconstructed, g_reconstructed, r_reconstructed)
)

diff = cv2.absdiff(img, img_reconstructed)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_reconstructed_rgb = cv2.cvtColor(
    img_reconstructed,
    cv2.COLOR_BGR2RGB
)
diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Compressed Image")
plt.imshow(img_reconstructed_rgb)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Difference Image")
plt.imshow(diff_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()
