import os
import daft
import re
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from uuid import uuid4
import json
from typing import List

from pyiceberg import catalog
from pyiceberg.schema import Schema
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.table.sorting import SortOrder, SortField, SortDirection
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    TimestamptzType,
    StringType,
    NestedField,
)
from pyiceberg.table import Table
import logging
import os

from fms_dgt.base.task import SdgData, SdgTask
import fms_dgt.utils as utils

lh_data_instance = None

root_log_level = utils.log_level
dmf_log_level = getattr(logging, os.getenv("DMF_LOG_LEVEL", "error").upper())

if root_log_level < dmf_log_level:
    #remove logs from libraries used for DMF integration
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(lambda record: not record.name.startswith("daft"))
        handler.addFilter(lambda record: not record.name.startswith("pyiceberg"))


class CosCredentials():
    """
    Abstraction to represent COS credentials
    """
    def __init__(self, key, secret, region=None, endpoint=None) -> None:
        if not key or not secret or not endpoint:
            raise ValueError("'key', 'secret' and 'endpoint' are mandatory arguments")
        self.key = key
        self.secret = secret
        self.region = region
        self.endpoint = endpoint

class NamespaceLocationCosKey():
    """
    Abstraction to represent Namespace location and cos credentials
    """
    def __init__(self, namespace: str, location: str, cos_credentials: CosCredentials) -> None:
        self.namespace = namespace
        self.location = location
        self.cos_credentials = cos_credentials

class LakehouseApi():

    def __init__(self, host, token) -> None:
        if not host or not token:
            raise ValueError("'host' and 'token' are required parameters")
        self.host = host
        self.token = token

    def _make_api_call(self, endpoint: str, params):
        """
        Generic Lakehouse API call template, given URL and params
        """
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        # sending post request and saving response as response object
        response = requests.get(url=f'{self.host}/{endpoint}', params=params, headers=headers) #, verify=False

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text) 
    
    def _get_lakehouse_user(
            self, 
        ) -> str:
        """
        Return the user id (email) based on the LH token
        """
        user =  self._make_api_call(endpoint=f"user", params={})
        return user.get('email')

    def _get_namespace_location_with_cos_credentials(self, namespace: str) -> NamespaceLocationCosKey:
        """
        Return the cos credentials to access the bucket for given location based on user authorization over the namespace.
        If user has the required permission access in the namespace will return proper credentials, otherwise an authorized response will be returned.
        In case, lakehouse doesn't have credentials for the location's bucket a bad request will be returned
        """
        params = {
            "namespaceName": namespace            
        }
        response =  self._make_api_call(endpoint=f"namespace_location_with_cos_credentials", params=params)
        credentials = response.get('cos_credentials')
        cosKeys = CosCredentials(credentials.get('hmac_access_key'), credentials.get('hmac_secret_key'), credentials.get('region'), credentials.get('endpoint'))
        return NamespaceLocationCosKey(namespace=namespace, location=response.get('location'), cos_credentials=cosKeys)


class Lakehouse():
        
    """
    Lakehouse abstraction
    """
    def __init__(self, namespace: str) -> None:
        load_dotenv()
        self.host = os.getenv("LAKEHOUSE_API", None)
        assert (
            self.host is not None
        ), f"Could not find API key 'LAKEHOUSE_API' in config or environment!"
        self.token = os.getenv("LAKEHOUSE_TOKEN", None)
        assert (
            self.token is not None
        ), f"Could not find API key 'LAKEHOUSE_TOKEN' in config or environment!"

        self.api = LakehouseApi(self.host,self.token)
        self.user_id = self.api._get_lakehouse_user()
        self.namespace = namespace

        ##load location and cos credentials
        namespace_details = self.api._get_namespace_location_with_cos_credentials(namespace=namespace)
        self.namespace_location = namespace_details.location
        self.cos = namespace_details.cos_credentials
        self._catalog = self._load_catalog()
        self.run_id = str(uuid4())
        #self.created_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') # Example format: 2024-05-07 18:10:00 (use UTC zone)
        self.created_at = datetime.now()

    def _load_catalog(self) -> catalog:
        return catalog.load_catalog(
            "cat_loader",
            **{
                "uri": self.host+ "/iceberg",
                "token": self.token,
                "s3.endpoint": self.cos.endpoint,
                "s3.access-key-id": self.cos.key,
                "s3.secret-access-key": self.cos.secret,
                "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
            },
        )
    

    def _get_input_table(self) -> Table:
        table_name = 'dgt_input'
        table_identifier = f'{self.namespace}.{table_name}'
        try:
            table = self._catalog.load_table(table_identifier)
            return table
        except NoSuchTableError:
            print("Table doesn't exist, creating...")

        schema = Schema(
            NestedField(field_id=1, name="run_id", field_type=StringType(), required=True),
            NestedField(field_id=2, name="user_id", field_type=StringType(), required=True),
            NestedField(field_id=3, name="task_name", field_type=StringType(), required=True),
            NestedField(field_id=4, name="data_builder", field_type=StringType(), required=False),
            NestedField(field_id=5, name="created_by", field_type=StringType(), required=False),
            NestedField(field_id=6, name="taxonomy_path", field_type=StringType(), required=False),
            NestedField(field_id=7, name="task_yaml", field_type=StringType(), required=False),
            NestedField(field_id=8, name="builder_config", field_type=StringType(), required=False),
            NestedField(field_id=9, name="created_at", field_type=TimestamptzType(), required=True),
            identifier_field_ids=[1,3]
        )
        # Sort on the generated_at
        sort_order = SortOrder(SortField(source_id=9, transform=IdentityTransform(), direction=SortDirection.DESC))
        #maintenance properties
        properties = {
                    'lh.maintenance-required' : 'true',
                    'lh.expire-snapshots-older-than-days' : '5',
                    'lh.rewrite-min-data-files' : '3',
                }

        self._catalog.create_table(
            identifier=table_identifier,
            schema=schema,
            location=f"{self.namespace_location}/{table_name}",
            properties=properties,
            sort_order=sort_order,
        )  

        table = self._catalog.load_table(table_identifier)
        return table

    def _normalize_table_name(self, table_name:str) -> str: 
        # Define the regular expression pattern for table name
        pattern = "^[a-z0-9._]+$"
        # Use re.match to check if the text matches the pattern
        if re.match(pattern, table_name):
            return table_name
        else:
            #TODO: improve it. Although seems all builder names are ok
            return table_name.replace("-", "_")

    def _get_output_table(self, data_builder: str, output: dict) -> Table:
        table_name = self._normalize_table_name(f'dgt_output_{data_builder}')
        table_identifier = f'{self.namespace}.{table_name}'
        try:
            table = self._catalog.load_table(table_identifier)
            return table
        except NoSuchTableError:
            print("Table doesn't exist, creating...")

        fields: list[NestedField] = []
        fields.append(NestedField(field_id=1, name="run_id", field_type=StringType(), required=True))
        fields.append(NestedField(field_id=2, name="user_id", field_type=StringType(), required=True))
        fields.append(NestedField(field_id=3, name="task_name", field_type=StringType(), required=True))
        fields.append( NestedField(field_id=4, name="created_at", field_type=TimestamptzType(), required=True))
        index = 5
        if "task_name" in output:
            # Remove the "age" attribute
            del output["task_name"]
        for key in output:
            fields.append( NestedField(field_id=index, name=key, field_type=StringType(), required=False))
            index += 1


        schema = Schema(
            fields=fields,
            identifier_field_ids=[1,3]
        )
        #maintenance properties
        properties = {
                    'lh.maintenance-required' : 'true',
                    'lh.expire-snapshots-older-than-days' : '5',
                    'lh.rewrite-min-data-files' : '3',
                }
        
        # Sort on the generated_at
        sort_order = SortOrder(SortField(source_id=4, transform=IdentityTransform(), direction=SortDirection.DESC))
        self._catalog.create_table(
            identifier=table_identifier,
            schema=schema,
            location=f"{self.namespace_location}/{table_name}",
            properties=properties,
            sort_order=sort_order,
        )  

        table = self._catalog.load_table(table_identifier)
        return table


    def save_task_details(self, task: SdgTask):
        #get the table
        table = self._get_input_table()

        #read the data file
        task_yaml = utils.read_data_file(task.file_path)
        for attr in utils.DATA_FILE_ADDITIONAL_ATTRIBUTES:
            del task_yaml[attr]

        #generate the record to save
        ret =  [{
                        "run_id": self.run_id,
                        "user_id": self.user_id,
                        "task_name": task.name,
                        "data_builder": task.data_builder,
                        "created_by": task._created_by,
                        "taxonomy_path": task.file_path,
                        "task_yaml": json.dumps(task_yaml),
                        "builder_config" : json.dumps(task.builder_cfg),
                        "created_at": self.created_at
                }]

        #save the record in lakehouse
        df = daft.from_pylist(ret)
        return df.write_iceberg(table)

    def save_task_data(self, task: SdgTask, new_data: List[SdgData]):
        example = new_data[0].to_output_dict()
        ret = []
        for d in new_data:
            output = d.to_output_dict()
            #generate the record to save
            for key, value in output.items():
                if not isinstance(value, str):
                    output[key] = json.dumps(value)
            
            ##add extra fields
            output["run_id"] = self.run_id
            output["user_id"] = self.user_id
            output["created_at"] = datetime.now(timezone.utc)
            if "task_name" not in output:
                output["run_id"] = task.name
            ret.append(output)

        #get the table
        table = self._get_output_table(data_builder=task.data_builder, output=example)
        
        #save the record in lakehouse
        df = daft.from_pylist(ret)
        return df.write_iceberg(table)
        
        
    