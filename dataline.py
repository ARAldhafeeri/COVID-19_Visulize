from pykafka import KafkaClient
import json
from datetime import datetime
import uuid
import time

input_file = open('data.json')
json_array = json.load(input_file)
coordinates = json_array["optimized_path_points"]
optimized_path_points = json_array["optimized_path_points"]
Confirmed = json_array["Confirmed"]
Dates = json_array["Dates"]
print(coordinates)

#kafka producer
client = KafkaClient(hosts="localhost:9092")
topic = client.topics['COVID-19-25']
producer = topic.get_sync_producer()



data = {}
data['busline'] = '00001'


def generate_checkpoint(coordinates,optimized_path_points,Confirmed, Dates):
    i = 0
    while i < len(coordinates):
        data['key'] = data['busline'] + '_' + str(i)+ str(uuid.uuid4())
        if (i < len(optimized_path_points)):
            data['optimized_path_points'] = optimized_path_points[i]
        data['latitude'] = optimized_path_points[i][0]
        data['longitude'] = optimized_path_points[i][1]
        if (i < len(Confirmed)):
            data['Confirmed'] = Confirmed[i] + Confirmed[i]+ Confirmed[i]+Confirmed[i]
        if( i < len(Dates)):
            data['Dates'] = Dates[i]
        i+=1
        message = json.dumps(data)
        print(message)
        producer.produce(message.encode('ascii'))
        time.sleep(.08)
        if i == len(coordinates)-1:
            i = 0
        else:
         i += 1
generate_checkpoint(coordinates,optimized_path_points,Confirmed, Dates)





