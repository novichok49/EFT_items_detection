from pathlib import Path
from typing import Dict, Tuple
from pycocotools.coco import COCO
class CustomCOCO:
    def __init__(self, path)->None:
        self.path = Path(path)
        self.__image_id = 0
        self.__annot_id = 0
        self.__categ_id = 0
        self.category_map = {}
        self.data = {'images': [],
                     'annotations': [],
                     'categories': []}
        
    def __getitem__(self, index):
        pass
        
    def add(self,
            image_name:str,
            size: Tuple,
            bboxes:Dict) -> None:
        image = {'id':self.__image_id,
                 'file_name': image_name,
                 'width': size[0],
                 'heigth': size[1]}
        self.data['images'].append(image)

        for class_name, bboxes in bboxes.items():
            if class_name in self.category_map:
                category_id = self.category_map[class_name]
            else:
                self.category_map[class_name] = self.__categ_id
                category_id
                self.__categ_id += 1

            category = {'id': category_id,
                        'name': class_name,
                        'supercategory': "none"}
            self.data['categories'].append(category)

            for bbox in bboxes:
                area = bbox[2] * bbox[3]
                segmentation = [bbox[0], bbox[1],
                                bbox[0], bbox[1] + bbox[3],
                                bbox[0] + bbox[2], bbox[1] + bbox[3],
                                bbox[0] + bbox[2], bbox[1]]
                # Annotation info for images bbox
                annotation = {'id': self.__annot_id,
                              'image_id': self.__image_id,
                              'area': area,
                              'bbox': bbox,
                              'segmentation': segmentation,
                              'iscrowd': 0,
                              'category_id': category['id']}
                self.data['annotations'].append(annotation)
                self.__annot_id += 1
        self.__image_id += 1
