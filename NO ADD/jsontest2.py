import json

with open('data.json') as json_file:
    data = json.load(json_file)
    for p in data['FeaturePoints']:
        for f in p['features']:
            print("Id_img: " , f['Px'])
        # print('Website: ' + p['website'])
        # print('From: ' + p['from'])
        # print('')