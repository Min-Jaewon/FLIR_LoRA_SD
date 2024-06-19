import os

import matplotlib.pyplot as plt
from PIL import Image

def create_and_save_grid_image(images, save_path, grid_x):
    grid_size=(grid_x, 1)
    img_width, img_height = images[0].size

    grid_img = Image.new('RGB', (img_width * grid_size[0], img_height * grid_size[1]))

    for i, img in enumerate(images):
        grid_x = i % grid_size[0] * img_width
        grid_y = i // grid_size[0] * img_height
        grid_img.paste(img, (grid_x, grid_y))

    if os.path.exists(save_path):
        print("Already Exist")
    else:
        print(f"{save_path} has been saved")
        grid_img.save(save_path)


result_path = 'result'
keywords = ['org', 'pag']

for keyword in keywords:
    image_list=[]
    
    for image in sorted(os.listdir(result_path)):   
        if keyword in image:
            print(f"{image} is added to the grid")
            image_path=os.path.join(result_path, image)
            image = Image.open(image_path)
            image_list.append(image)
    
    save_path = f"{keyword}_grid" + '.png'
    num_image=len(image_list)
    create_and_save_grid_image(image_list, save_path, num_image)

