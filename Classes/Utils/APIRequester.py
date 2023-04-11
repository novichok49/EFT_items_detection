import requests
from graphql_query import Query
from typing import List, Dict

API_URL = 'https://api.tarkov.dev/graphql'
HEADERS = {"Content-Type": "application/json"}


class APIRequester:
    def __init__(self) -> None:
        self._last_response = None

    def request(self, name: str, fields: List[str]) -> List[Dict]:
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
