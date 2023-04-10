from PIL import Image
from pathlib import Path
from io import BytesIO
import requests
from Classes.Utils import APIRequester
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
        """
        Constructor method for ImagesDownloader class.

        Args:
            path (str | Path): _description_
            link_fields (List[str]): _description_

        Raises:
            Exception: _description_
        """
        if not all(field in IMG_LINK_FIELDS for field in link_fields):
            raise Exception("Bad field name in link_fields.")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.images_path = path
        self.image_fields = link_fields
        self.im_id = 0
        self.images_data = self.__get_image_links()
        self.failed_download = {key: [] for key in self.image_fields}

    def __get_image_links(self) -> List[dict]:
        """
        Method to get items data from API tarkov.dev.

        Returns:
            List[dict]: API response incude items info.
        """
        fields = ['id']
        fields.extend(self.image_fields)
        API = APIRequester()
        response = API.request(name='items', fields=fields)
        return response

    def download(self):
        """
        Dowload images from links in response.
        """
        for field in self.image_fields:
            save_path = Path(self.images_path, field[:-4])
            save_path.mkdir(exist_ok=True)
            for item in self.images_data:
                link = item[field]
                response = requests.get(link)
                if response.status_code == 200:
                    item_id = item['id']
                    image_id = self.im_id
                    image = Image.open(
                        BytesIO(response.content)).convert(mode='RGBA')
                    image_name = f'{item_id}.{image_id}.png'
                    image.save(save_path / image_name)
                    self.im_id += 1
                else:
                    self.failed_download[field].append(link)
