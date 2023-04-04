from typing import Optional

from vespa.application import ApplicationPackage
from vespa.deployment import Vespa, VespaDocker

from amazon_product_search.vespa.package import create_package


def start(name_or_id: Optional[str] = None, app_package: Optional[ApplicationPackage] = None) -> Vespa:
    """Start the Vespa container

    Args:
        name_or_id (Optional[str], optional): The container name or ID..
        app_package (Optional[ApplicationPackage], optional): The ApplicationPackage to deploy.
            When it is not given, ApplicationPackage(name="amazon") is used.

    Returns:
        Vespa: The Vespa app.
    """
    if name_or_id:
        vespa_docker = VespaDocker.from_container_name_or_id(name_or_id)
    else:
        vespa_docker = VespaDocker()

    if not app_package:
        app_package = create_package()

    vespa_app = vespa_docker.deploy(application_package=app_package)
    return vespa_app


def stop(name_or_id: str = "amazon"):
    """Stop the Vespa container.

    Args:
        name_or_id (str, optional): The container name or ID. Defaults to "amazon".
    """
    vespa_docker = VespaDocker.from_container_name_or_id(name_or_id)
    vespa_docker.container.stop()


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
    if not app_package:
        app_package = create_package()

    vespa_app = Vespa(host, application_package=app_package)
    vespa_app.wait_for_application_up(max_wait=50)
    return vespa_app
