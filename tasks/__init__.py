from invoke import Collection

from tasks import data_tasks, demo_tasks, es_tasks, gcloud_tasks, synonyms_tasks, vespa_tasks

ns = Collection()
ns.add_collection(Collection.from_module(data_tasks, name="data"))
ns.add_collection(Collection.from_module(demo_tasks, name="demo"))
ns.add_collection(Collection.from_module(es_tasks, name="es"))
ns.add_collection(Collection.from_module(gcloud_tasks, name="gcloud"))
ns.add_collection(Collection.from_module(synonyms_tasks, name="synonyms"))
ns.add_collection(Collection.from_module(vespa_tasks, name="vespa"))
