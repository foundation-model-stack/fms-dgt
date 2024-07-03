# Standard
from typing import Dict

# First Party
from fms_dgt.base.registry import register_resource
from fms_dgt.base.resource import BaseResource


@register_resource("gpu")
class GpuResource(BaseResource):
    def __init__(self, device: str):
        super().__init__(device)
        self._device = device

    @property
    def device(self):
        return self._device
