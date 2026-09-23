import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt

T = 1000
beta = np.linspace(1e-4 , 0.02 , T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size,size))
    return np.asarray(img , np.float32) / 255 * 2 - 1

def apply_noise(x0, t , rng):
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps

def to_display(x):
   return np.clip((x + 1) / 2 , 0 ,1)

path = sys.argv[1]
rng = np.random.default_rng(0)
x0 = load_image(path , 255)

fig , axes = plt.subplots(1 ,5, figsize=(3 * 5 , 3.4))
steps = [i for i in range(0 , 500 , 100)]
print(steps)
for ax , t in zip(axes, steps):
    xt = apply_noise(x0 , t , rng)
    ax.imshow(to_display(xt))
    print(f"step T {t} , Mean : {xt.mean()} , Variance : {xt.var()}")

plt.show()
