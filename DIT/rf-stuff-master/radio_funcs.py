import numpy as np
import math as m
import pandas as pd
from scipy import optimize

'''

my_loc is an array which has the lat and long coords of interested locations

1.  calc_dist -- this function requires two arguments: each argument is an array which has two values stored in it, a latitude and longitude. This function returns the distance between the two provided points

2.  calc_rec_power -- this function requires 5 arguemnts: a distance value, the transimtted power in dBm, the transmit antenna gain, the receive antenna gain, the transmission frequency, and whether you want Watts or dBm back. From there it returns the received power in dBm or Watts

3.  dist_matrix -- this function requires one argument: an array of location values (containing lat & long). It will return a matrix which displays the distance between each location stored

4.  rec_power_matrix -- this function requires one argument: an array of distance values. It returns the a matrix of power received values separated by the distances provided in the distance matrix

5.  all_in_one -- this function requires one argument: an array of location values (lat & long). It combines function number 3 and 4 and has the same output as function number 4 

6.  all_in_one_db -- this function requires one argument: an array of location values (lat & long). It combines function number 3 and 4, but constructs a pandas database of the received power at each pad site

'''









#################----------------- Store all the pad sites lat and long coords ----------------#######################

my_loc = [[33.357941, -111.539376],[33.352782, -111.540084],[33.357604, -111.524427],[33.360122, -111.522646],[33.363289, -111.527447],[33.354664, -111.552356]] # [Latitude, Longitude]

pad_names = []
for i in range(1, len(my_loc)+1):
	pad_names.append('Location {}'.format(i))

###########------------------------- Calculate the distance between two points ---------------##########################

def calc_dist(v1,v2):
	v1 = np.radians(v1) # Convert the lat/long coord to radians
	v2 = np.radians(v2) # Convert the lat/long coord to radians
	lat1 = v1[0]
	long1 = v1[1]

	lat2 = v2[0]
	long2 = v2[1]

	deltaLat = lat2 - lat1
	deltaLong = long2 - long1

	R = 6371e3 # radius of the Earth (meters)

	var1 = ((np.sin(deltaLat/2.))**2) + (np.cos(lat1)*np.cos(lat2)*(np.sin(deltaLong/2.)**2))

	var2 = 2*(np.arctan2(m.sqrt(var1), np.sqrt(1-var1)))

	return(R*var2) # meters

############--------------------------- Calculate the Received Power (Using Frii's Free space equation)--------------------------############################

def calc_rec_power(dist, pt, gt, gr, f, units):
	Pt = 10**((pt-30)/10) # transmitted power in Watts
	lam = 3e8/f # our signal wavelength in meters

	constant = Pt*gt*gr*(lam/(4*np.pi))**2
	if dist == 0:
		return(np.inf)
	else:
		power_W = constant/(dist**2) # Recevied Power in Watts

		power_dBm = 10*np.log10(power_W) + 30 # Received Power in dBm
	if units == 'watts':
		return(power_W)
	elif units == 'dbm':
		return(power_dBm)

########-------------------- Construct a matrix which stores all the pads distances between one another -------------########

def dist_matrix(pad_loc):
	distance_matrix = []
	for i in range(0, len(pad_loc)):
		dist_list = []
		for j in range(0, len(pad_loc)):
			dist_list.append(calc_dist(pad_loc[i], pad_loc[j]))
		distance_matrix.append(dist_list)
	return(distance_matrix)

#########------------------ Construct a matrix which stores the received power values between all pad sites -----------######

def rec_power_matrix(distance_matrix, pt, gt, gr, f, units):
	received_power = []
	for i in range(0, len(distance_matrix)):
		power_list = []
		for j in range(0, len(distance_matrix[i])):			
			if distance_matrix[i][j] < 1:				
				power_list.append(0)
			else:
				power_list.append(calc_rec_power(distance_matrix[i][j], pt, gt, gr, f, units))
		received_power.append(power_list)
	return(received_power)

#######------------------- Combine dist_matrix & rec_power_matrix functions into one ------------------#################

def all_in_one(pad_loc):
	distance_matrix = []
	for i in range(0, len(pad_loc)):
		dist_list = []
		for j in range(0, len(pad_loc)):
			dist_list.append(calc_dist(pad_loc[i], pad_loc[j]))
		distance_matrix.append(dist_list)
	for i in range(0, len(distance_matrix)):
		for j in range(0, len(distance_matrix[i])):
			if distance_matrix[i][j] == 0.0:
				distance_matrix[i][j] = 1e-8
	received_power = []
	for i in range(0, len(distance_matrix)):
		power_list = []
		for j in range(0, len(distance_matrix)):	
			power_list.append(calc_rec_power(distance_matrix[i][j]))
		received_power.append(power_list)
	return(received_power)

#######------------------------- Return the power received between each pad site as a pandas database ------------##########

def database(pad_loc):
	distance_matrix = []
	for i in range(0, len(pad_loc)):
		dist_list = []
		for j in range(0, len(pad_loc)):
			dist_list.append(calc_dist(pad_loc[i], pad_loc[j]))
		distance_matrix.append(dist_list)
	received_power = []
	for i in range(0, len(distance_matrix)):
		power_list = []
		for j in range(0, len(distance_matrix)):	
			power_list.append(calc_rec_power(distance_matrix[i][j]))
		received_power.append(power_list)
	
	received_power = pd.DataFrame(received_power)

	received_power.columns = pad_names

	received_power.insert(0, 'Pad #', pad_names)
	
	return(received_power)	


def test_func(r, a, b):
	return(a*(np.power(r, b)))



