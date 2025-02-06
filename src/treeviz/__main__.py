from treeviz import cli

parser = cli.build_parser()
args = parser.parse_args()
cli.run_cli(args)
