from PIL import Image
from pathlib import Path
from io import BytesIO
import requests
from Classes.Utils import APIRequester, ImagesDir
from typing import List

# Fields contains link to image download
IMG_LINK_FIELDS = ["iconLink", "gridImageLink",
                   "baseImageLink", "image8xLink",
                   "inspectImageLink", "image512pxLink"]


class ImagesDownloader():
    """
    Class for download images from tarkov.dev API.
    """

    def __init__(self, path: str | Path, link_fields: List[str]) -> None:
        if not all(field in IMG_LINK_FIELDS for field in link_fields):
            raise Exception("Bad field name in link_fields.")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.images_path = path
        self.image_fields = link_fields
        self.images_data = self.__get_image_links()
        self.failed_download = {key: [] for key in self.image_fields}

    def __get_image_links(self) -> List[dict]:
        fields = ['name']
        fields.extend(self.image_fields)
        API = APIRequester()
        response = API.request(name='items', fields=fields)
        return response

    def download(self):
        """
        Dowload images from links in response.
        """
        im_dir = ImagesDir(self.images_path)
        for field in self.image_fields:
            for item in self.images_data:
                link = item[field]
                response = requests.get(link)
                if response.status_code == 200:
                    item_class = item['name']
                    image = Image.open(
                        BytesIO(response.content)).convert(mode='RGBA')
                    im_dir.add_image(image, item_class)
                else:
                    self.failed_download[field].append(link)
        im_dir.save_info()
