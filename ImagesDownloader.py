from PIL import Image
from pathlib import Path
from io import BytesIO
from typing import List
from os import mkdir
import pandas as pd
import requests


class ImagesDownloader():
    def run_query(query: str) -> pd.DataFrame:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            url='https://api.tarkov.dev/graphql',
            headers=headers,
            json={'query': query})
        if response.status_code == 200:
            response = response.json()
            norm_response = pd.json_normalize(response['data']['items'])
            return pd.DataFrame(norm_response)
        else:
            raise Exception("Query failed to run by returning code of {}. {}".format(
                response.status_code, query))

    def download_images(
            df: pd.DataFrame,
            name_col: str,
            columns: List[str],
            save_path: Path) -> pd.DataFrame:
        for column in columns:
            column_name = f'{column}.filepath'
            mkdir(save_path / column_name)
            for id, item in df.iterrows():
                item_name = item[name_col]
                item_image_link = item[column]
                bimg = requests.get(item_image_link).content
                img = Image.open(BytesIO(bimg)).convert(mode='RGBA')
                filepath = f'{column_name}/{item_name}.png'
                df.loc[id, column_name] = filepath
                img.save(save_path / filepath)
        return df
