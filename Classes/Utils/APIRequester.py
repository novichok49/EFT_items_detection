from requests.exceptions import RequestException
import requests
from graphql_query import Query, Field
from typing import List, Dict
import pandas as pd

# TODO Refactor all class


class APIRequester:
    """
    The APIRequester class that allows you to make
    API requests to api.tarkov.dev.
    """
    API_URL: str = 'https://api.tarkov.dev/graphql'
    HEADERS: Dict = {"Content-Type": "application/json"}

    @classmethod
    def post(cls, name: str, fields: List[str]) -> List[Dict]:
        """
        A class method that allows you to make a POST request to the
        API with the given parameters.

        Arguments:
            `name` -- Representing the object name from api.tarkov.dev.\n
            `fields` -- Representing the fields you want to retrieve from
                name object.

        Returns:
            Response data in json format.
        """
        query = Query(name=name, fields=fields)
        query = f'{{{query.render()}}}'
        response = requests.post(
            url=cls.API_URL,
            headers=cls.HEADERS,
            json={'query': query})
        if response.status_code == 200:
            response = response.json()['data'][name]
            return response
        else:
            raise RequestException(f"Request failed \
                with status code {response.status_code}")

    def get_flea_items(self) -> pd.DataFrame:
        """
        Get list all items which  can sale on flea

        Returns:
            Pandas dataframe with items can sell on flea
        """
        query = Query(name='items',
                      fields=['normalizedName',
                              Field(name='sellFor',
                                    fields=[Field(name='vendor',
                                                  fields=['normalizedName'])])])
        query = f'{{{query.render()}}}'
        response = requests.post(
            url=APIRequester.API_URL,
            headers=APIRequester.HEADERS,
            json={'query': query})
        if response.status_code == 200:
            response = response.json()['data']
            response = pd.json_normalize(data=response['items'],
                                         record_path='sellFor',
                                         meta='normalizedName')
            return response[response['vendor.normalizedName'] == 'flea-market'][['normalizedName']]
        else:
            raise RequestException(f"Request failed \
                with status code {response.status_code}")
