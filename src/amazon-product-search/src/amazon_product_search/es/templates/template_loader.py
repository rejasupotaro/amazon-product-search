from jinja2 import Environment, FileSystemLoader, Template
from importlib import resources


class TemplateLoader:
    def __init__(self) -> None:
        package_path = str(resources.files("amazon_product_search"))
        search_path = f"{package_path}/es/templates"
        file_system_loader = FileSystemLoader(searchpath=search_path)
        self.environment = Environment(loader=file_system_loader)

    def load(self, template_name: str) -> Template:
        return self.environment.get_template(template_name)
