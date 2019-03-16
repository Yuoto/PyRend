import json

def saveJson(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def loadJson(path):
    with open(path, 'r') as f:
        return json.load(f)