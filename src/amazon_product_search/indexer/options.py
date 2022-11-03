from apache_beam.options.pipeline_options import PipelineOptions


class IndexerOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument("--index_name", type=str)
        parser.add_argument("--locale", type=str)
        parser.add_argument("--es_host", type=str)
        parser.add_argument("--extract_keywords", action="store_true")
        parser.add_argument("--encode_text", action="store_true")
        parser.add_argument("--nrows", type=int, default=-1)
