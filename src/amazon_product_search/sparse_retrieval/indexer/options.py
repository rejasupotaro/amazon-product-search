from apache_beam.options.pipeline_options import PipelineOptions


class IndexerOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument("--locale", type=str)
        parser.add_argument("--es_host", type=str)
        parser.add_argument("--nrows", type=int, default=-1)
