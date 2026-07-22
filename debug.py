from pprint import pprint
from parse.parser import load_yaml, build_config, validate_config

yaml_path = "cfg/example.yaml"

raw_config = load_yaml(yaml_path)
config = build_config(raw_config, yaml_path)
validate_config(config)

pprint(config, sort_dicts=False)