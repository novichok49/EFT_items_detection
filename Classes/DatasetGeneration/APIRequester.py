import requests
from graphql_query import Query
from typing import List, Dict

API_URL = 'https://api.tarkov.dev/graphql'
HEADERS = {"Content-Type": "application/json"}


class APIRequester:
    """A class for making requests
    to a tarkov.dev API returning the response data
    in a Python dictionary format.
    """

    def __init__(self) -> None:
        self._last_response = None

    def request(self, name: str, fields: List[str]) -> Dict:
        query = Query(name=name, fields=fields)
        query = f'{{{query.render()}}}'
        response = requests.post(
            url=API_URL,
            headers=HEADERS,
            json={'query': query})
        self._last_response = response.json()['data'][name]
        if response.status_code == 200:
            return self._last_response
        else:
            raise Exception("Query failed to run by returning code of {}. {}".format(
                response.status_code, query))

    @property
    def last_response(self):
        return self._last_response
