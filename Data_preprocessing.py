# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:41:03 2020

@author: iahme
"""

# Import china covid-19 time line update
import pandas as pd
data = pd.read_csv('COVID-19-china.csv')

# Get number of clumns to clean data and remove final column
len(data.columns)
# Delete last column index 91
data.drop(data.columns[91], axis=1, inplace=True)

# First thing we need coordinates which are in index 2 and index 3
# (index 2, index 3) >>> (Lat, Log)
lat = data.iloc[:, 2].values
log = data.iloc[:,3].values

All_dates = data.columns
All_dates.drop(['Country/Region'])
All_dates.drop(index=['Lat'])
All_dates = All_dates.tolist()
All_dates = All_dates[3:]
All_dates = All_dates[1:]


# Converting lat, log into lists
list_lat = lat.tolist()
list_log = log.tolist()

# Generating  original coordinates
Original_coorndinates = [(list_lat[i], list_log[i]) for i in range(0, len(list_log))] 


"Generate a list of confirmed casses  for every original coordinate"

# Get list of confirmed cases coordinate 0
Original_coor_0_confirmed = data.iloc[0,4:91].values
Original_coor_0_confirmed = Original_coor_0_confirmed.tolist()

# Get list of confirmed cases coordinate 1
Original_coor_1_confirmed = data.iloc[1,4:91].values
Original_coor_1_confirmed = Original_coor_1_confirmed.tolist()

# Get list of confirmed cases coordinate 2
Original_coor_2_confirmed = data.iloc[2,4:91].values
Original_coor_2_confirmed = Original_coor_2_confirmed.tolist()

# Get list of confirmed cases coordinate 3
Original_coor_3_confirmed = data.iloc[3,4:91].values
Original_coor_3_confirmed = Original_coor_3_confirmed.tolist()

# Get list of confirmed cases coordinate 4
Original_coor_4_confirmed = data.iloc[4,4:91].values
Original_coor_4_confirmed = Original_coor_4_confirmed.tolist()

# Get list of confirmed cases coordinate 5
Original_coor_5_confirmed = data.iloc[5,4:91].values
Original_coor_5_confirmed = Original_coor_5_confirmed.tolist()
# Get list of confirmed cases coordinate 6
Original_coor_6_confirmed = data.iloc[6,4:91].values
Original_coor_6_confirmed = Original_coor_6_confirmed.tolist()

# Get list of confirmed cases coordinate 7
Original_coor_7_confirmed = data.iloc[7,4:91].values
Original_coor_7_confirmed = Original_coor_7_confirmed.tolist()

# Get list of confirmed cases coordinate 8
Original_coor_8_confirmed = data.iloc[8,4:91].values
Original_coor_8_confirmed = Original_coor_8_confirmed.tolist()

# Get list of confirmed cases coordinate 9
Original_coor_9_confirmed = data.iloc[9,4:91].values
Original_coor_9_confirmed = Original_coor_9_confirmed.tolist()

# Get list of confirmed cases coordinate 10
Original_coor_10_confirmed = data.iloc[10,4:91].values
Original_coor_10_confirmed = Original_coor_10_confirmed.tolist()

# Get list of confirmed cases coordinate 11
Original_coor_11_confirmed = data.iloc[11,4:91].values
Original_coor_11_confirmed = Original_coor_11_confirmed.tolist()


# Get list of confirmed cases coordinate 12
Original_coor_12_confirmed = data.iloc[12,4:91].values
Original_coor_12_confirmed = Original_coor_12_confirmed.tolist()

# Get list of confirmed cases coordinate 13
Original_coor_13_confirmed = data.iloc[13,4:91].values
Original_coor_13_confirmed = Original_coor_13_confirmed.tolist()

# Get list of confirmed cases coordinate 14
Original_coor_14_confirmed = data.iloc[14,4:91].values
Original_coor_14_confirmed = Original_coor_14_confirmed.tolist()

# Get list of confirmed cases coordinate 15
Original_coor_15_confirmed = data.iloc[15,4:91].values
Original_coor_15_confirmed = Original_coor_15_confirmed.tolist()

# Get list of confirmed cases coordinate 16
Original_coor_16_confirmed = data.iloc[16,4:91].values
Original_coor_16_confirmed = Original_coor_16_confirmed.tolist()

# Get list of confirmed cases coordinate 17
Original_coor_17_confirmed = data.iloc[17,4:91].values
Original_coor_17_confirmed = Original_coor_17_confirmed.tolist()
# Get list of confirmed cases coordinate 18
Original_coor_18_confirmed = data.iloc[18,4:91].values
Original_coor_18_confirmed = Original_coor_18_confirmed.tolist()

# Get list of confirmed cases coordinate 19
Original_coor_19_confirmed = data.iloc[19,4:91].values
Original_coor_19_confirmed = Original_coor_19_confirmed.tolist()

# Get list of confirmed cases coordinate 20
Original_coor_20_confirmed = data.iloc[20,4:91].values
Original_coor_20_confirmed = Original_coor_20_confirmed.tolist()

# Get list of confirmed cases coordinate 21
Original_coor_21_confirmed = data.iloc[21,4:91].values
Original_coor_21_confirmed = Original_coor_21_confirmed.tolist()

# Get list of confirmed cases coordinate 22
Original_coor_22_confirmed = data.iloc[22,4:91].values
Original_coor_22_confirmed = Original_coor_22_confirmed.tolist()


# Get list of confirmed cases coordinate 23
Original_coor_23_confirmed = data.iloc[23,4:91].values
Original_coor_23_confirmed = Original_coor_23_confirmed.tolist()

# Get list of confirmed cases coordinate 24
Original_coor_24_confirmed = data.iloc[24,4:91].values
Original_coor_24_confirmed = Original_coor_24_confirmed.tolist()

# Get list of confirmed cases coordinate 25
Original_coor_25_confirmed = data.iloc[25,4:91].values
Original_coor_25_confirmed = Original_coor_25_confirmed.tolist()

# Get list of confirmed cases coordinate 26
Original_coor_26_confirmed = data.iloc[26,4:91].values
Original_coor_26_confirmed = Original_coor_26_confirmed.tolist()

# Get list of confirmed cases coordinate 27
Original_coor_27_confirmed = data.iloc[27,4:91].values
Original_coor_27_confirmed = Original_coor_27_confirmed.tolist()

# Get list of confirmed cases coordinate 28
Original_coor_28_confirmed = data.iloc[28,4:91].values
Original_coor_28_confirmed = Original_coor_28_confirmed.tolist()
# Get list of confirmed cases coordinate 29
Original_coor_29_confirmed = data.iloc[29,4:91].values
Original_coor_29_confirmed = Original_coor_29_confirmed.tolist()

# Get list of confirmed cases coordinate 30
Original_coor_30_confirmed = data.iloc[30,4:91].values
Original_coor_30_confirmed = Original_coor_30_confirmed.tolist()

# Get list of confirmed cases coordinate 31
Original_coor_31_confirmed = data.iloc[31,4:91].values
Original_coor_31_confirmed = Original_coor_31_confirmed.tolist()

# Get list of confirmed cases coordinate 32
Original_coor_32_confirmed = data.iloc[32,4:91].values
Original_coor_32_confirmed = Original_coor_32_confirmed.tolist()



"""  
Generate New imagniary coordinate based on:
  
  New infected most likely was 6 feets way from containgous person
  somewhere around the patient 0 in coordinate 0. 
  6 feets are (0.000008, 0.000008) away in form of (west, north) or ( log, lat)
  
  in This model we are using (0.000040, 0.000040)
  
  Also we take into the account the confirmed casses generate a number of points
  based on  the confirmed casses of every day for that city.

 """





def generate_child(confirmed_cases,original_coordinate,
                                  N=3,
                                  new_patient = (0.001208,0.001208),
                                  coordinate_0_childrens = []):
    """
    confirmed_cases: Number of confirmed cases through time from day 0
      Tuple
      expect(numbers of confirmed cases integer)
      
    original_coordinate: The original coordinate of the city or case 1
      expect (x,y) table.
      
    coordinate_0_childrens: The predicted locations that the virus might 
    traveled in.
      list of tuples ( x,y)
    new_patient: New covid patient most luckly came acroos the virus 6 feets away
    which meens (0.000008,0.000008) , (West, North)  or ( log, lat)
      
    N: optional arugment to set number of predicted locations
    integer
    
    """

    coordinate_0_childrens = []
    new_coordinates = tuple()
    for confirmed_case in Original_coor_0_confirmed:
      if(  confirmed_case > 0):
        for i in range(0, N):
          increment =(0.000940,0.00140)
          new_coordinates = tuple(map(sum,zip(original_coordinate,new_patient)))
          coordinate_0_childrens.append(new_coordinates)
          new_patient = tuple(map(sum,zip(increment,new_patient)))

    return coordinate_0_childrens
          
          
# Generate prediction of location for childreen of coordinate 0

children_0 = generate_child(Original_coor_0_confirmed,Original_coorndinates[0])

      
# Same for rest.

children_1 = generate_child(Original_coor_1_confirmed,Original_coorndinates[1])
  



children_2 = generate_child(Original_coor_2_confirmed,Original_coorndinates[2])



children_3 = generate_child(Original_coor_3_confirmed,Original_coorndinates[3])




children_4 = generate_child(Original_coor_4_confirmed,Original_coorndinates[4])
  



children_5 = generate_child(Original_coor_5_confirmed,Original_coorndinates[5])




children_6 = generate_child(Original_coor_6_confirmed,Original_coorndinates[6])




children_7 = generate_child(Original_coor_7_confirmed,Original_coorndinates[7])




children_8 = generate_child(Original_coor_8_confirmed,Original_coorndinates[8])



children_9 = generate_child(Original_coor_9_confirmed,Original_coorndinates[9])



children_10 = generate_child(Original_coor_10_confirmed,Original_coorndinates[10])



children_11 = generate_child(Original_coor_11_confirmed,Original_coorndinates[11])



children_12 = generate_child(Original_coor_12_confirmed,Original_coorndinates[12])



children_13 = generate_child(Original_coor_13_confirmed,Original_coorndinates[13])


children_14 = generate_child(Original_coor_14_confirmed,Original_coorndinates[14])



children_15 = generate_child(Original_coor_15_confirmed,Original_coorndinates[15])


children_16 = generate_child(Original_coor_16_confirmed,Original_coorndinates[16])



children_17 = generate_child(Original_coor_17_confirmed,Original_coorndinates[17])



children_18 = generate_child(Original_coor_18_confirmed,Original_coorndinates[18])



children_19 = generate_child(Original_coor_19_confirmed,Original_coorndinates[19])


children_20 = generate_child(Original_coor_20_confirmed,Original_coorndinates[20])


children_21 = generate_child(Original_coor_21_confirmed,Original_coorndinates[21])




children_22 = generate_child(Original_coor_22_confirmed,Original_coorndinates[22])



children_23 = generate_child(Original_coor_23_confirmed,Original_coorndinates[23])



children_24 = generate_child(Original_coor_24_confirmed,Original_coorndinates[24])



children_25 = generate_child(Original_coor_25_confirmed,Original_coorndinates[25])




children_26 = generate_child(Original_coor_26_confirmed,Original_coorndinates[26])



children_27 = generate_child(Original_coor_27_confirmed,Original_coorndinates[27])


children_28 = generate_child(Original_coor_1_confirmed,Original_coorndinates[28])



children_29 = generate_child(Original_coor_1_confirmed,Original_coorndinates[29])



children_30 = generate_child(Original_coor_1_confirmed,Original_coorndinates[30])


children_31 = generate_child(Original_coor_31_confirmed,Original_coorndinates[31])

children_32 = generate_child(Original_coor_32_confirmed,Original_coorndinates[32])


# Now we add all the children togather to generate all the possible points 
# That the viurs might came acroos.


Preprocessed_Coordinates  = ( children_0 + children_1 + children_2 + children_3 
                            + children_4 + children_5 + children_6 
                            + children_7 + children_8  + children_9 
                            + children_10 + children_11 + children_12
                            + children_13 + children_14 + children_15 
                            + children_16+ children_17 + children_18 
                            + children_19 + children_20 + children_21
                            + children_22 + children_23 + children_24
                            + children_25 + children_26 + children_27
                            + children_28 + children_29 + children_30
                            + children_31 + children_32)

All_confirmed_cases = (Original_coor_0_confirmed
                     +Original_coor_1_confirmed
                     +Original_coor_2_confirmed
                     +Original_coor_3_confirmed
                     +Original_coor_4_confirmed
                     +Original_coor_5_confirmed
                     +Original_coor_6_confirmed
                     +Original_coor_7_confirmed
                     +Original_coor_8_confirmed
                     +Original_coor_9_confirmed
                     +Original_coor_10_confirmed
                     +Original_coor_11_confirmed
                     +Original_coor_12_confirmed
                     +Original_coor_13_confirmed
                     +Original_coor_14_confirmed
                     +Original_coor_15_confirmed
                     +Original_coor_16_confirmed
                     +Original_coor_17_confirmed
                     +Original_coor_18_confirmed
                     +Original_coor_19_confirmed
                     +Original_coor_20_confirmed
                     +Original_coor_21_confirmed
                     +Original_coor_22_confirmed
                     +Original_coor_23_confirmed
                     +Original_coor_24_confirmed
                     +Original_coor_25_confirmed
                     +Original_coor_26_confirmed
                     +Original_coor_27_confirmed
                     +Original_coor_28_confirmed
                     +Original_coor_29_confirmed
                     +Original_coor_30_confirmed    
                     +Original_coor_31_confirmed
                     +Original_coor_32_confirmed
    
    )


# Creating Hashmap of the data


"Here we are applying traveling salesperson problem algorithm & vislizing it"
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
data_array = np.array(Preprocessed_Coordinates)
# Change Size of(data_array)
selected_from_data_array = data_array[np.random.choice(data_array.shape[0],  
                                            replace=False,  
                                            size=8613)]
# Get distances from coordinates x,y
distances = pdist(selected_from_data_array)
# Create distance matrix
distances_matrix = squareform(distances)

from tsp_solver.greedy_numpy import solve_tsp

# Get the optimized path from slove_tsp
optimized_path = solve_tsp(distances_matrix) 
# get optimized_path_points
optimized_path_points = [selected_from_data_array[x] for x in optimized_path]

list_optimized_path_points = []

for item in optimized_path_points:
  convert =item.tolist()
  list_optimized_path_points.append(convert)
dict1 = dict()

dict1["Coordinates"] = Preprocessed_Coordinates
dict1["optimized_path"] = optimized_path
dict1["optimized_path_points"] = list_optimized_path_points
dict1["Confirmed"] = All_confirmed_cases
dict1["Dates"] = All_dates

a = str(dict1)

import json

with open('data.json','w') as json_file:
    json.dump(a, json_file)
#
# plt.figure(figsize=(10, 10), dpi=300)
# plt.plot([x[1] for x in optimized_path_points],
#          [x[0] for x in optimized_path_points],
#          color='black', lw=1)
# plt.xlim(0, 100)
# plt.ylim(0, 200)
# plt.gca().invert_yaxis()
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# from tspy import TSP
#
# tsp = TSP()
# tsp.read_mat(distances_matrix)
# tsp.read_data(data_array)
#
# from tspy.solvers import TwoOpt_solver
# two_opt = TwoOpt_solver(initial_tour='NN', iter_num=131)
# two_opt_tour = tsp.get_approx_solution(two_opt)
# tsp.plot_solution('TwoOpt_solver')
#
          
