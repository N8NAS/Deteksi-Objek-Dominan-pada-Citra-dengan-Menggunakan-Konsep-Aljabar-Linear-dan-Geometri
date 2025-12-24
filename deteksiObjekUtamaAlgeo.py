import numpy as np
import math
from PIL import Image

# Fungsi-fungsi
def mean(M):
    s = 0.0
    cnt = 0
    for row in M:
        for x in row:
            s += x
            cnt += 1
    return s / cnt
def euclidean(v1, v2):
    s = 0.0
    for i in range(len(v1)):
        d = v1[i] - v2[i]
        s += d * d
    return math.sqrt(s)
def svd(A, k):
    ATA = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eig(ATA)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    U, S, V = [], [], []

    for i in range(k):
        lambda_i = eigenvalues[i]
        if lambda_i < 1e-9:
            break
        sigma = math.sqrt(lambda_i)
        v = eigenvectors[:, i]
        u = (A @ v) / (sigma + 1e-9)
        U.append(u)
        S.append(sigma)
        V.append(v)
    return np.array(U), np.array(S), np.array(V)

img = Image.open("image.jpg").convert("L")
A = np.asarray(img, dtype=float)
h, w = A.shape
A_list = A.tolist()
A_mean = mean(A_list)
A_centered = A - A_mean

k = 8
U, S, V = svd(A_centered, k)

features = np.zeros((h, w, k))
for i in range(len(S)):
    features[:, :, i] = S[i] * np.outer(U[i], V[i])

F = features.reshape(-1, k)
centroid = [0.0] * k

for j in range(k):
    for i in range(F.shape[0]):
        centroid[j] += F[i][j]
    centroid[j] /= F.shape[0]

centroid = np.array(centroid)
dist = []
for i in range(F.shape[0]):
    dist.append(euclidean(F[i], centroid))

dist = np.array(dist)
dist_img = dist.reshape(h, w)
threshold = sum(dist) / len(dist)
mask = dist_img < threshold

objek1 = np.zeros_like(A)
objek2 = np.zeros_like(A)
objek1[mask] = A[mask]
objek2[~mask] = A[~mask]

Image.fromarray(np.array(objek1, dtype=np.uint8)).save("objek lain.png")
Image.fromarray(np.array(objek2, dtype=np.uint8)).save("objek dominan.png")

print("Selesai: Deteksi Objek Utama")



#def normalize_image(M):
#    flat = [x for row in M for x in row]
##    mn = min(flat)
#    mx = max(flat)
#    return [
#        [int(255 * (x - mn) / (mx - mn + 1e-9)) for x in row]
#        for row in M
#    ]
#objek1_norm = normalize_image(objek1.tolist())
#objek2_norm = normalize_image(objek2.tolist())
