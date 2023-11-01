from requests.exceptions import RequestException
import requests
from graphql_query import Query, Field
from typing import Dict, List


class ApiService:
    """
    Service for make queries to https://api.tarkov.dev/.
    """
    API_URL: str = 'https://api.tarkov.dev/graphql'
    HEADERS: Dict = {"Content-Type": "application/json"}

    @staticmethod
    def __convert_query(query: List[Dict | str]) -> Query:
        """
        Convert query tree to graph QL `Query` type. 

        Arguments:
            query -- Query with `str` fields and `dict` subfields.

        Raises:
            TypeError: If query elem not `dict` or `str` type.

        Returns:
            Converted graphQL `Query`. 
        """
        converted_query = []
        for elem in query:
            if isinstance(elem, dict):
                name = list(elem.keys())[0]
                fields = ApiService.__convert_query(elem[name])
                converted_query.append(Field(name=name, fields=fields))
                return converted_query
            elif isinstance(elem, str):
                converted_query.append(elem)
            else:
                raise TypeError(f'Elem {elem} not support.')
        return converted_query

    @staticmethod
    def items(fields: List[Dict | str]) -> List | Dict:
        """
        Post query to api for get items information.
        See Item in https://api.tarkov.dev/.

        Arguments:
            fields -- `str` fields and `dict` subfields.

        Raises:
            RequestException: Request failed.

        Returns:
            `List` of query results or `dict` with errors.
        """
        name = 'items'
        query = ApiService.__convert_query(fields)
        query = Query(name='items', fields=query)
        query = f'{{{query.render()}}}'
        response = requests.post(
            url=ApiService.API_URL,
            headers=ApiService.HEADERS,
            json={'query': query})
        if response.status_code == 200:
            response = response.json()
            if 'data' in response: 
                response = response['data'][name]
            else:
                raise Exception(f'Request failed \
                with {response["errors"]}.')
            return response
        else:
            raise RequestException(f"Request failed \
                with code {response.status_code}.")