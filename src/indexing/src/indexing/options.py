from apache_beam.options.pipeline_options import PipelineOptions


class IndexerOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser) -> None:
        parser.add_argument("--pipeline_type", type=str)
        parser.add_argument("--locale", type=str)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--nrows", type=int, default=-1)
        parser.add_argument("--source", type=str, default="file")
        parser.add_argument("--encode_text", action="store_true")
        parser.add_argument("--dest", type=str, default="stdout")
        parser.add_argument("--dest_host", type=str)
        parser.add_argument("--index_name", type=str)
        parser.add_argument("--table_id", type=str)
