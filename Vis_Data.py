from Data_preprocessing import Myprocessed_coordinates, Myoptimized_path, Myoptimized_path_points

import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key='AIzaSyAOqp3DWLZwlA_vUjmWM6iGObkjR4Ytft0')

# Geocoding an address
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions("Sydney Town Hall",
                                     "Parramatta, NSW",
                                     mode="transit",
                                     departure_time=now)




