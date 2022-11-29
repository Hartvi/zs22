import numpy as np
import math as m
import pandas as pd
from radio_funcs import database, my_loc, pad_names, dist_matrix, rec_power_matrix, test_func,calc_dist,calc_rec_power
import matplotlib.pyplot as plt
from scipy import optimize

plt.rc('text', usetex = False) # allows us to make use of LaTeX fonts and script writing
plt.rc('font', family = 'serif')

distances = dist_matrix(my_loc)

pt = 50
gt = 12
gr = 12
units = 'watts'
f = 1.773e9

rec_power = rec_power_matrix(distances, pt, gt, gr, f, units)

#########----------------- Sort our distance values by least to greatest ----------------###############

sort_dist = []
for i in range(0,len(distances)):
	x = sorted(distances[i])
	sort_dist.append(x)


#####---------------------- Sort our power values by least to greatest -----------------------##########

sort_power = []
for i in range(0,len(rec_power)):
	x = sorted(rec_power[i], reverse=True)
	sort_power.append(x)

#####------------------------ In our case, 0 values are artificial data points and need to be excluded from the dataset --------------------#############

remove_zero = []
for i in range(0,len(sort_dist)):	
	no_zeros = []	
	for j in range(0, len(sort_dist)):
		if sort_dist[i][j] != 0.0:		
			no_zeros.append(sort_dist[i][j])
	remove_zero.append(no_zeros)

dist_max_min = []
for i in remove_zero:
	for j in i:
		dist_max_min.append(j)


remove_zero_power = []
for i in range(0,len(sort_power)):	
	no_zeros_power = []	
	for j in range(0, len(sort_power)):
		if sort_power[i][j] != 0.0:		
			no_zeros_power.append(sort_power[i][j])
	remove_zero_power.append(no_zeros_power)

max_min_list = []
for i in remove_zero_power:
	for j in i:
		max_min_list.append(j)
		

##### To add a best fit line first we need to create a function we wish to test. Friis equation goes as 1/r^2 and so it is safe to guess a solution as r^b #####
#def test_func(r, a, b):
	#return(a*(np.power(r, b)))



a_values = []
b_values = []
for i in range(0, len(sort_power)):
	if len(remove_zero[i]) != len(remove_zero_power[i]):
		pass	
	else:
		params, params_covariance = optimize.curve_fit(test_func, remove_zero[i], remove_zero_power[i])
		plt.scatter(remove_zero[i], remove_zero_power[i])
		plt.plot(remove_zero[i], test_func(remove_zero[i], params[0], params[1]))
		a_values.append(params[0])
		b_values.append(params[1])


a_average = np.average(a_values) # calculates the average value found for a over all plotted datasets
b_average = np.average(b_values) # calculates the average value found for b over all plotted datasets
e = "Fitted solution is $ar^{b}$"
a = ('a average: {}'.format(a_average))
b = ('b average: {}'.format(b_average))
c = ('Transmitted Power = {} W'.format(10**((pt-30)/10)))
#plt.legend([c, e, a, b], loc = 'best', fontsize = 20) # adds a legend which displays our guess function and the average values calculated for a and b
plt.text(0.5*max(dist_max_min), 0.5*max(max_min_list), 'Transmitted Power = {} W'.format(10**((pt-30)/10)), fontsize=14)
plt.ylim([1.1*min(max_min_list), 1.1*max(max_min_list)])
plt.xlabel('Distance from Tx (m)', fontsize = 22)
plt.ylabel('Received Power (W)', fontsize = 22)
plt.title('Power Received vs. Distance from Tx', fontsize = 22)
plt.show()

