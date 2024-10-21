# 把标准正太分布生成张量，然后还原为一张图片
# 加噪声的过程，最终原始图片会变成一张接近标准正态分布张量所生成的图片

import torch
import torchvision
import matplotlib.pyplot as plt

# 生成标准正态分布的张量
tensor = torch.randn(1000, 1000)

# 将张量转换为图像
image = torchvision.transforms.functional.to_pil_image(tensor)

# 显示图像
plt.imshow(image, cmap='gray')
plt.show()
