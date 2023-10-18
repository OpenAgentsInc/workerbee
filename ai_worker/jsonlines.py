import json


def load_jsonlines(fin):
    while True:
        lin = fin.readline()
        yield json.loads(lin)

