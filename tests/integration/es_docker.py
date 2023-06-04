import json
from json.decoder import JSONDecodeError
from time import sleep
from typing import cast

import docker
from docker.errors import NotFound
from docker.models.containers import Container


class EsDocker:
    def __init__(self, container_id: str):
        docker_client = docker.from_env()
        try:
            container = docker_client.containers.get(container_id=container_id)
            if not isinstance(container, Container):
                raise TypeError("The type of container must be Container")
            self.container = cast(Container, container)
            self.container.restart()
        except NotFound:
            self.container = docker_client.containers.run(
                image="docker.elastic.co/elasticsearch/elasticsearch:8.7.0",
                name=container_id,
                ports={9200: 9200},
                environment={
                    "xpack.security.enabled": "false",
                    "discovery.type": "single-node",
                },
                detach=True,
            )
        self.wait_for_init()

    def wait_for_init(self):
        status = ""
        waited, max_wait = 0, 30
        while waited <= max_wait:
            sleep(5)
            waited += 5
            output = self.container.exec_run(
                "bash -c 'curl -s http://localhost:9200/_cluster/health'"
            ).output.decode("utf-8")
            try:
                response = json.loads(output)
            except JSONDecodeError:
                continue
            status = response["status"]
            if status != "green":
                continue
            break
        if status != "green":
            raise Exception("Failed to start Elasticsearch")

    def stop(self):
        self.container.stop()
        self.container.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
