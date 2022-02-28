
import os
import argparse
from torchvision import transforms as transforms
from PIL import Image
import torch
import test_model as TestModel
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='unko')
    parser.add_argument('--load_model', type=str, default='./latest_net_G.pth', help='unko')
    parser.add_argument('--load_img', type=str, default='./img[110]_real.png', help='unko')
    parser.add_argument('--out_img', type=str, default='./img[110]_fake.png', help='unko')
    parser.add_argument('--input_nc', type=int, default=3, help='unko')
    parser.add_argument('--output_nc', type=int, default=3, help='unko')
    parser.add_argument('--ngf', type=int, default=64, help='unko')
    opt = parser.parse_args()
#    print(opt)
    model = TestModel.TestModel(opt)

    totensor = transforms.ToTensor()
    PIL = transforms.ToPILImage()
    data = Image.open(opt.load_img)
    data = data.convert('RGB')
    width  = data.width
    height = data.height
    data = data.resize((256, 256), Image.LANCZOS)
    data = torch.stack([(totensor(data)-0.5)*2],dim=0)
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    visuals[0] = torch.clamp(visuals[0]/2+0.5,0,1)
    result = PIL(visuals[0])
    result = result.resize((math.floor(512*(width/height)), 512), Image.LANCZOS)
#    result = result.resize((512, 512), Image.LANCZOS)
    result.save(opt.out_img)

