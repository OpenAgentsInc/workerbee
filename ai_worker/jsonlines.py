import json


def load_jsonlines(fin):
    while True:
        lin = fin.readline()
        if not lin:
            return
        yield json.loads(lin)

