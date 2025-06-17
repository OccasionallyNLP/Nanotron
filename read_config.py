import yaml

with open('examples/config_smollm_qwen.yaml') as f:
    x = yaml.load(f, Loader=yaml.FullLoader)

print(x['data_stages'][0]['data']['dataset'])
print(type(x['data_stages'][0]['data']['dataset'][0]))