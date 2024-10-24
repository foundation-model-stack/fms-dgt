# Local
import os
from dotenv import load_dotenv
from fms_dgt.base.registry import register_resource
from fms_dgt.resources.api import ApiKeyResource


@register_resource("watsonx")
class WatsonXResource(ApiKeyResource):

    def __init__(
        self,
    ):
        super().__init__(key_name="WATSONX_API_KEY", call_limit=10)

        # Load configuration
        load_dotenv()

        # Initialize WatsonX Project ID
        self._project_id = os.getenv("WATSONX_PROJECT_ID", None)
        if not self._project_id:
            raise AssertionError("WATSONX_PROJECT_ID environment variable must be set.")

        # Initialize WatsonX API url
        self._url = os.getenv("WATSONX_API_URL", "https://us-south.ml.cloud.ibm.com")

        # Initialize WatsonX API token, if requested
        self._token = os.getenv("WATSONX_API_TOKEN", None)

    @property
    def project_id(self):
        return self._project_id

    @property
    def url(self):
        return self._url

    @property
    def token(self):
        return self._token
