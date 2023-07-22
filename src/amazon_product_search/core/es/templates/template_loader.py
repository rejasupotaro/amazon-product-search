from typing import Optional

from jinja2 import Environment, FileSystemLoader, Template

from amazon_product_search.constants import PROJECT_DIR


class TemplateLoader:
    def __init__(self, searchpath: Optional[str] = None):
        if not searchpath:
            searchpath = f"{PROJECT_DIR}/src/amazon_product_search/core/es/templates"
        file_system_loader = FileSystemLoader(searchpath=searchpath)
        self.environment = Environment(loader=file_system_loader)

    def load(self, template_name: str) -> Template:
        return self.environment.get_template(template_name)
