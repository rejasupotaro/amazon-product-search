from jinja2 import Environment, FileSystemLoader, Template

from amazon_product_search.constants import PROJECT_DIR


class TemplateLoader:
    def __init__(self, project_dir: str = PROJECT_DIR):
        searchpath = f"{project_dir}/src/amazon_product_search/core/es/templates"
        file_system_loader = FileSystemLoader(searchpath=searchpath)
        self.environment = Environment(loader=file_system_loader)

    def load(self, template_name: str) -> Template:
        return self.environment.get_template(template_name)
