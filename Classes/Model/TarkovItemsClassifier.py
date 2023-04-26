import torch
import torchvision


class TarkovItemsClassifier:
    def __init__(self):
        pass

def create_anchor_generator_from_w_h(w_resize=1024 / 1920):
    from Classes.Utils import APIRequester
    import pandas as pd
    from torchvision.models.detection.rpn import AnchorGenerator
    
    api = APIRequester()
    response = api.post('items', ['height', 'width'])
    #Create dataframe from api response
    df = pd.json_normalize(data=response)

    df = df.groupby(['height', 'width']).agg(lambda x: 0)
    # это соотношения всех якорей
    anchors_per_elem = df.index.to_list()
    # Количество якорей для каждого пикселя
    df = pd.DataFrame(anchors_per_elem, columns=['h', 'w'])
    df['ratio'] = df['h'] / df['w']
    df = df.groupby('h').apply(lambda x: x['ratio'].values)

    # Aspect ratios
    aspect_ratios = []
    for elem in df:
        aspect_ratios.append(list(elem))
    #TODO Переписать универсально
    sizes = [[64], [128], [192], [256], [320], [384], [448]]
    for s in sizes:
        s[0] = s[0] * w_resize

    return AnchorGenerator(sizes=sizes,
                             aspect_ratios=aspect_ratios)