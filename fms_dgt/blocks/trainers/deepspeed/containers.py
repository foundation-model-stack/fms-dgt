# Standard
import io
import logging
import subprocess
import tarfile

# Third Party
import docker


class Container:
    def __init__(self, container_name: str, container_file: str, container_dir: str):
        """Create container from provided file and run it in specified directory

        Args:
            container_name (str): docker image name to instantiate
            container_file (str): file to build docker from
            container_dir (str): where to run
        """
        self._container_name = container_name

        try:
            client = docker.from_env()
        except docker.docker.DockerException as de:
            d_p = "podman"
            init_command = (
                f"{d_p} build -t {container_name} -f {container_file} {container_dir}"
            )
            subprocess.run(init_command, shell=True)
            client = docker.from_env()

        self._container = client.containers.create(
            self._container_name, "sleep infinity"
        )
        self._container.start()
        logging.info(
            f"Container using image {self._container_name} has now started. Info: {self._container}"
        )

    def run_cmd(self, cmd: str):
        self._container.run(cmd)
