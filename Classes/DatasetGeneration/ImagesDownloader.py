from PIL import Image
from pathlib import Path
from io import BytesIO
import pandas as pd
import requests

# Columns contains link to image download
IMG_LINK_COLUMNS = ["iconLink", "gridImageLink", "baseImageLink"]
# QUERY to tarkov.dev API 
QUERY="""
{
  items {
    id
    normalizedName
    width
    height
    iconLink
    gridImageLink
    baseImageLink
  }
}
"""

class ImagesDownloader():
    """
    Class for download images from tarkov.dev API.
    """
    def __init__(self, path: str | Path='./Images') -> None:
        """
        Constructor method for ImagesDownloader class.

        Parameters:
        - path: str or Path object representing the directory where images will be saved.
          Default is './Images'.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.images_path = path
        self._images_types = IMG_LINK_COLUMNS
        self.failed_download = {key: [] for key in self._images_types}
        self._fetch_images_data()
        self._download_images()

    def _fetch_images_data(self):
        """
        Method to fetch images data from API tarkov.dev and store it as a Pandas DataFrame.
        """
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            url='https://api.tarkov.dev/graphql',
            headers=headers,
            json={'query': QUERY})
        if response.status_code == 200:
            response = response.json()
            norm_response = pd.json_normalize(response['data']['items'])
            norm_response = norm_response.set_index("id")
            self.images_data =  norm_response
        else:
            raise Exception("Query failed to run by returning code of {}. {}".format(
                response.status_code, QUERY))
        
    def _download_images(self):
        """
        Method to download images and save them to disk. Skip download if file existst.
        """
        for column in self._images_types:
            save_path = Path(self.images_path, column)
            save_path.mkdir(exist_ok=True)
            for id, row in self.images_data.iterrows():
                if (save_path / f"{id}.png").exists():
                    continue
                response = requests.get(row[column])
                if response.status_code == 200:
                    byte_img = response.content
                    img = Image.open(BytesIO(byte_img)).convert(mode='RGBA')
                    img.save(save_path / f"{id}.png")
                else:
                    self.failed_download[column].append(row[column])
