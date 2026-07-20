import yaml
config = yaml.load(open('cfg/example.yaml', 'r'), Loader=yaml.SafeLoader)
print(config)