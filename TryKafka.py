import pykafka as kafka
import json
from datetime import datetime

client = kafka.KafkaClient(hosts="localhost:9092")

while True:
    message = ("heelo"+ str(count))
    kafka.Producer(message)
    print(message)
    count = count + 1

input_file = open('data.json')
json_array = json.load(input_file)
coordinates = json_array["optimized_path_points"]
print(coordinates)

data = {}
data['line'] = '000001'

def generate_checkpoint(coordinates):
    i = 0
    while i < len(coordinates):
        data['key'] = data['line'] + '_' + str(i)
        data['timestamp'] = str(datetime.utcnow())
        data['latitude'] = coordinates[i][1]
        data['logitude'] = coordinates[i][0]
        i+=1
        message = json.dumps(data)
        print(message)
generate_checkpoint(coordinates)