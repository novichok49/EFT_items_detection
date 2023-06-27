from pathlib import Path
from typing import Tuple, Dict
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class BaseImages:
    """
    Class for managing images and their associated labels.
    """
    def __init__(self, path: str | Path) -> None:
        """
        Initializes a new instance of BaseImages.

        Args:
            `path` -- The path to the directory
            where images and labels will be stored.
        """
        self.dir_path = Path(path)
        if not self.dir_path.exists():
            self.dir_path.mkdir(exist_ok=True)
        self.data = self.load_state()
        self.last_image_id = len(self.data)
        self.need_encode = True

    def __getitem__(self, image_id: slice | int) -> Tuple[Image.Image, int]:
        """
        Retrieves the image and label information
        for the given image ID or slice.

        Arguments:
            `image_id` -- slice or int image index.

        Raises:
            TypeError: If the provided `image_id` type is unsupported.

        Returns:
            A tuple containing the image and its associated label ID.
        """
        if self.need_encode:
            self.create_labels()
            self.need_encode = False

        if isinstance(image_id, slice):
            rows = self.data[image_id.start:image_id.stop:image_id.step]
            items_list = []
            for tuple in rows.itertuples(index=True):
                filename = tuple[0]
                class_id = tuple[1]
                if pd.isna(class_id):
                    class_id = None
                image_path = self.dir_path / f"{filename}.png"
                image = Image.open(image_path)
                items_list.append((image, class_id))
            return items_list
        elif isinstance(image_id, int):
            row = self.data.iloc[image_id]
            filename = row.name
            class_id = row["id"]
            if pd.isna(class_id):
                class_id = None
            image_path = self.dir_path / f"{filename}.png"
            image = Image.open(image_path)
            return image, class_id
        elif isinstance(image_id, str):
            row = self.data[self.data['label'] == image_id]
            filename = row.index[0]
            class_id = row['id'][0]
            if pd.isna(class_id):
                class_id = None
            image_path = self.dir_path / f"{filename}.png"
            image = Image.open(image_path)
            return image, class_id
        else:
            raise TypeError(f'Unsupported index type: {type(image_id)}.')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.__getitem__(self.index)
            self.index += 1
            return result
        else:
            raise StopIteration

    def __del__(self) -> None:
        self.save_state()

    def add_image(self,
                  image: Image.Image | Path,
                  class_name: str,
                  visible: bool=True) -> None:
        """
        Adds an image with its associated class name and visibility.

        Arguments:
            `image` -- The PIL image or path to the image file.\n
            `class_name` -- The class name associated with the image.\n

        Keyword Arguments:
            `visible` -- The visibility status of the `image`. (default: {True})

        Raises:
            TypeError: If the provided image argument type is unsupported.
        """
        image_id = str(self.last_image_id).zfill(6)
        save_name = f'{image_id}.png'

        if isinstance(image, Image.Image):
            image.save(self.dir_path / save_name)
        elif isinstance(image, Path):
            image.replace(self.dir_path / save_name)
        else:
            raise TypeError(f'Unsupported arg type {type(image)}')
        self.need_encode = True
        self.data.loc[image_id] = {"label": class_name,
                                   "visible": visible}
        self.last_image_id += 1

    def create_labels(self, sep: int=0) -> None:
        """
        Creates label encoding for visible images.

        Keyword Arguments:
            `sep` -- The separation value to add to the encoded labels.
            If you need start labeling from 1 set sep to 1. (default: {0})
        """
        indexs = self.data["visible"] == True
        labels = self.data.loc[indexs, "label"]
        ids = LabelEncoder().fit_transform(labels)
        self.data["id"] = pd.Series(dtype=pd.Int64Dtype(), data=None)
        self.data.loc[indexs, "id"] = ids + sep
        

    def set_visible_classes(self, visible: Dict[str, bool]) -> None:
        """
        Sets the visibility of classes.

        Arguments:
            `visible` -- A dictionary mapping class
            names to their visibility status.
        """
        fc = lambda row: visible[row['label']]
        self.data['visible'] = self.data.apply(fc, axis=1)

    def get_decode_map(self) -> Dict[int, str]:
        """
        Retrieves a dictionary mapping encoded index
        to their corresponding class names.

        Returns:
            A dictionary mapping encoded index to class names.
        """
        if self.need_encode:
            self.create_labels()
            self.need_encode = False
        visible = self.data["visible"] == True
        decode_map = {k:v for k, v in zip(self.data.loc[visible, "id"],
                                          self.data.loc[visible, "label"])}
        return decode_map

    def get_encode_map(self) -> Dict[str, int]:
        """
        Retrieves a dictionary mapping class names
        to their corresponding encoded index.

        Returns:
            A dictionary mapping class names to encoded index.
        """
        if self.need_encode:
            self.create_labels()
            self.need_encode = False
        visible = self.data["visible"] == True
        encode_map = {v:k for k, v in zip(self.data.loc[visible, "id"],
                                          self.data.loc[visible, "label"])}
        return encode_map

    def save_state(self) -> None:
        """
        Saves the data to csv file.
        """
        self.data[["label", "visible"]].to_csv(self.dir_path / ".csv")

    def load_state(self) -> pd.DataFrame:
        """
        Load the imagename, class label, and class visibility
        from a CSV file.

        Returns:
            The loaded label data as a pandas DataFrame.
        """
        label_path = self.dir_path / '.csv'
        if label_path.exists():
            im_data = pd.read_csv(label_path, index_col=0, dtype={"image": str})
        else:
            im_data = pd.DataFrame(columns=["image",
                                            "label",
                                            "visible"])\
                .set_index("image")
        return im_data