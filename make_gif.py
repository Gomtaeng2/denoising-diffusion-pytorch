import os
from PIL import Image

folder_path = 'results/'

image_list = []
for i in range(25):
    file_name = f'sample-71_step_{i}.png'
    file_path = os.path.join(folder_path, file_name)
    img = Image.open(file_path)
    image_list.append(img)

gif_path = os.path.join('gif/', 'sample-71.gif')
image_list[0].save(gif_path, save_all=True, append_images=image_list[1:], duration=100, loop=0)
