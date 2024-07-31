import torch
import SamStegNet
import random
import torchvision
from torchvision.utils import make_grid, save_image


manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

module_conceal = SamStegNet.Module_conceal().cuda()
module_reveal = SamStegNet.Module_reveal().cuda()

module_conceal.load_state_dict(torch.load("Module_conceal.pth"), strict=False)
module_reveal.load_state_dict(torch.load("Module_reveal.pth"), strict=False)

def get_loader(dataset_path,batch_size=5,size=64,shuffle=True):
    ## 数据预处理
    data_transfrom=torchvision.transforms.Compose([
        torchvision.transforms.Resize((size,size)),
        torchvision.transforms.RandomHorizontalFlip(),      # 对图片进行随机水平翻转
        torchvision.transforms.ToTensor(),                  # 将PIL格式，数组格式转换为tensor格式
                                                   ])
    ## 加载数据并进行预处理
    data_set=torchvision.datasets.ImageFolder(dataset_path,data_transfrom)
    ## 构造数据迭代器
    data_loader=torch.utils.data.DataLoader(data_set,batch_size=batch_size,shuffle=shuffle,num_workers=1)
    return data_loader

fixed_noise = torch.randn(5,100,1,1).cuda()


def main():
    secret_loader = get_loader("./test_images/", batch_size=5, size=64)
    for secret_images, _ in secret_loader:
        secret_images = secret_images.cuda()

    stego = module_conceal(fixed_noise,secret_images)
    recovery = module_reveal(stego)

    grid = make_grid(stego, nrow=5, padding=2, normalize=True, range=(0, 1))
    save_image(grid, 'stego.png')

    grid = make_grid(recovery, nrow=5, padding=2, normalize=True, range=(0, 1))
    save_image(grid, 'recovery.png')


if __name__ == '__main__':
    main()