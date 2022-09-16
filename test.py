import torch
import ssim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.io import read_image
from ssim import mySSIM
from skimage.metrics import structural_similarity as skSSIM

def tensor2numpy(img):
    return np.array(img.cpu().detach())

# Test Bench #1
print("Test #1")
print("Calculate  Skimage    Benchmark")

benchmark = [1.000, 0.988, 0.913, 0.840, 0.694, 0.662]

originalImg = read_image("data\image4.jpg") / 255.
inputImg = originalImg.unsqueeze(0)

for i in range(6):
    testImg = read_image("data\image{}.jpg".format(4+2*i)) / 255.
    outputImg = testImg.unsqueeze(0)
    calcSSIM = mySSIM(inputImg, outputImg).item()
    testSSIM = skSSIM(tensor2numpy(originalImg), tensor2numpy(testImg), channel_axis=0)
    print("{:.3f}      {:.3f}      {:.3f}".format(calcSSIM, testSSIM, benchmark[i]))

print()
# Test Bench #2
print("Test #2")
print("Calculate  Skimage    Benchmark")

originalImg = read_image("data\simga_0_ssim_1.0000.png") / 255.
inputImg = originalImg.unsqueeze(0)

testImg = read_image("data\simga_0_ssim_1.0000.png") / 255.
outputImg = testImg.unsqueeze(0)

calcSSIM = mySSIM(inputImg, outputImg).item()
testSSIM = skSSIM(tensor2numpy(originalImg), tensor2numpy(testImg), channel_axis=0)
print("{:.4f}     {:.4f}     1.0000".format(calcSSIM, testSSIM))

testImg = read_image("data\simga_50_ssim_0.4225.png") / 255.
outputImg = testImg.unsqueeze(0)

calcSSIM = mySSIM(inputImg, outputImg).item()
testSSIM = skSSIM(tensor2numpy(originalImg), tensor2numpy(testImg), channel_axis=0)
print("{:.4f}     {:.4f}     0.4225".format(calcSSIM, testSSIM))

testImg = read_image("data\simga_100_ssim_0.1924.png") / 255.
outputImg = testImg.unsqueeze(0)

calcSSIM = mySSIM(inputImg, outputImg).item()
testSSIM = skSSIM(tensor2numpy(originalImg), tensor2numpy(testImg), channel_axis=0)
print("{:.4f}     {:.4f}     0.1924".format(calcSSIM, testSSIM))