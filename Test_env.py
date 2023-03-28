from PIL import Image
from pathlib import Path
from ImagesDownloader import ImagesDownload
import matplotlib.pyplot as plt
import pandas as pd
from ImagesGenerator import ImagesGenerator


IMAGES_DATA_PATH = Path('./Tarkov_items.csv')
IMG_ID = 521


items_df = pd.read_csv(IMAGES_DATA_PATH)


image_preparing = ImagesGenerator(images_data=items_df,
                                  column='image512pxLink.filepath')

image_preparing.rescale_grid(h_grid='height',
                             w_grid='width',
                             save_path=Path('./images/'))
