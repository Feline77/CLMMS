import RRTS as turtle_hiding
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 生成秘密数据
def generate_binsecret(secret_length=3):
    bin_secret=[]
    for _ in range(secret_length):
        bin_secret.append(np.random.randint(0,2))
    return bin_secret

secret_length = 90000
secret = generate_binsecret(secret_length=secret_length)

# 读入载体图像
cover = cv2.imread("./example_images/4.png",cv2.IMREAD_GRAYSCALE)
cover = cv2.resize(cover,(256,256))
plt.imshow(cover,cmap="gray")


turtle_seed = 6
turtle_size = 256
stego,action_mat = turtle_hiding.hiding(cover,secret,turtle_seed,turtle_size)


turtle_seed = 6
turtle_size = 256
recovered_secret,recovered_image = turtle_hiding.extracting(stego,action_mat,secret_length,turtle_seed,turtle_size)
