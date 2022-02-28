import json
import glob
import os

for x in glob.glob('../ref/*.json'):
    with open(x) as f:
        data = json.load(f)
    for k, v in data.items():
        del v['entity']
        del v['relation']
        data[k] = v

    file = os.path.basename(x)
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)
