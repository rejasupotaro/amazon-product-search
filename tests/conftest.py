from apache_beam.testing.test_pipeline import TestPipeline

# This is to suppress
#   PytestCollectionWarning: cannot collect test class 'TestPipeline' because
#   it has a __init__ constructor (from: tests/amazon_product_search/indexer/io/test_elasticsearch_io.py)
# See: https://adamj.eu/tech/2020/07/28/how-to-fix-a-pytest-collection-warning-about-web-tests-test-app-class/
TestPipeline.__test__ = False
