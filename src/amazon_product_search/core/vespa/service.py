from typing import Optional

from docker.models.containers import Container

from amazon_product_search.core.vespa.package import create_package
from vespa.application import ApplicationPackage
from vespa.deployment import Vespa, VespaDocker


def start(
    name_or_id: Optional[str] = None, app_package: Optional[ApplicationPackage] = None
) -> Vespa:
    """Start the Vespa container

    Args:
        name_or_id (Optional[str], optional): The container name or ID..
        app_package (Optional[ApplicationPackage], optional): The ApplicationPackage to deploy.
            When it is not given, ApplicationPackage(name="amazon") is used.

    Returns:
        Vespa: The Vespa app.
    """
    vespa_docker = (
        VespaDocker.from_container_name_or_id(name_or_id)
        if name_or_id
        else VespaDocker()
    )
    if app_package:
        return vespa_docker.deploy(app_package)
    else:
        return vespa_docker.deploy(create_package())


def stop(name_or_id: str = "amazon") -> None:
    """Stop the Vespa container.

    Args:
        name_or_id (str, optional): The container name or ID. Defaults to "amazon".
    """
    vespa_docker = VespaDocker.from_container_name_or_id(name_or_id)
    vespa_docker.stop_services()
    container = vespa_docker.container
    if isinstance(container, Container):
        container.stop()


def connect(host: str, app_package: Optional[ApplicationPackage] = None) -> Vespa:
    """Connect to an existing Vespa app.

    Args:
        host (str): The hostname of the Vespa app to connect.
        app_package (Optional[ApplicationPackage], optional): The ApplicationPackage to deploy.
            When it is not given, ApplicationPackage(name="amazon") is used.

    Returns:
        Vespa: The Vespa app.
    """
    if not host:
        host = "http://localhost:8080"

    if app_package:
        vespa_app = Vespa(host, application_package=app_package)
    else:
        vespa_app = Vespa(host, application_package=create_package())

    vespa_app.wait_for_application_up(max_wait=50)
    return vespa_app
