import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

nb_experiments = 5
# Serial implementation using HD1 and HD2
nb_iteration = 4000   
x = {2,3,4,5,6}  		
nb_good_matches = [176, 1848, 6531, 14173, 23153]
time_serial = [0.220413, 0.815872, 2.711111, 5.282763, 8.310717]


# Parallel OMP_NUM_THREADS=7 ./feature_matcher_sift_parallel HD1.JPG HD2.JPG 4000 3
#nb_good_matches = 176
nb_threads = [2, 3, 4, 5, 6, 7, 8]
time_thread_176 = [0.122223, 0.110822, 0.098107, 0.097736, 0.098435, 0.098833, 0.099309]
time_thread_1848 = [0.440161, 0.435061, 0.423810, 0.422768, 0.426571, 0.426588, 0.428311]
time_thread_6531 = [1.329538, 1.339314, 1.323086, 1.342137]
time_thread_14173 = []
time_thread_23153 = []

0.428311
plt.plot(nb_good_matches, time_serial)
plt.xlabel("Number of good matches")
plt.ylabel("Time in secs")
plt.title("Serial RANSAC")
plt.show()