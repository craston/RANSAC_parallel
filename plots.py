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
x = {2,3,4,5,6,7,8,9}  		
nb_good_matches = [176, 1848, 6531, 14173, 23153]
time_serial = [0.220413, 0.815872, 2.711111, 5.282763, 8.310717]

# plt.figure()
# plt.plot(nb_good_matches, time_serial)
# plt.xlabel("Number of good matches with number of iteration = 4000")
# plt.ylabel("Time in secs")
# plt.title("Serial RANSAC (Time v/s number of good matches)")
# plt.show()

# nb_good_matches = 3
# nb_iteration = [1000, 2000, 3000, 4000, 5000]
time_serial = [0.200607, 0.402738, 0.609317, 0.797939, 1.018781]

# plt.figure()
# plt.plot(nb_iteration, time_serial)
# plt.xlabel("Number of iteration with number of good matches = 1848")
# plt.ylabel("Time in secs")
# plt.title("Serial RANSAC (Time v/s number of iterations)")
# plt.show()

#Parallel increasing threads
nb_iteration = 1000
nb_good_matches = 23153
nb_threads = [1,2,4,8,16]
time_parallel = [2.031124, 1.108913, 1.103770, 1.092959, 1.100372]
plt.figure()
plt.plot(nb_threads, time_parallel)
plt.xlabel("Number of threads with number of good matches = 23153, and number of iteration = 1000")
plt.ylabel("Time in secs")
plt.title("Parallel RANSAC (Time v/s number of threads)")
plt.show()

nb_iteration = [1000, 2000, 3000, 4000, 5000]
nb_good_matches = 1848
nb_threads = [2]
time_parallel = [0.109668, 0.220741, 0.328543, 	0.438512, 0.551060]
plt.figure()
plt.plot(nb_iteration, time_parallel , label="Parallel RANSAC")
plt.plot(nb_iteration, time_serial, label="Serial RANSAC")
plt.xlabel("Number of iteration with number of good matches = 1848, and number of threads = 2")
plt.ylabel("Time in secs")
plt.title("Parallel RANSAC Speedup (Time v/s number of iteration)")
plt.legend()
plt.show()

plt.figure()
plt.plot(nb_iteration, time_serial)
plt.xlabel("Number of iteration with number of good matches = 1848 and number of threads = 2")
plt.ylabel("Time in secs")
plt.title("Serial RANSAC (Time v/s number of iterations)")
plt.show()

# Parallel OMP_NUM_THREADS=7 ./feature_matcher_sift_parallel HD1.JPG HD2.JPG 4000 3
#nb_good_matches = 176
nb_threads = [2, 3, 4, 5, 6, 7, 8]
time_thread_176 = [0.122223, 0.110822, 0.098107, 0.097736, 0.098435, 0.098833, 0.099309]
time_thread_1848 = [0.440161, 0.435061, 0.423810, 0.422768, 0.426571, 0.426588, 0.428311]
time_thread_6531 = [1.329538, 1.339314, 1.323086, 1.342137, 1.337385, 1.323601, 1.331483]
time_thread_14173 = [2.815552, 2.788621, ]
time_thread_23153 = [4.628215, 4.746940]



#For 2 threads
stastic = [4.936876, 4.616397, 4.366911]
Dyanmic = [4.893106, 4.309880, 4.503628]  
Guided  = [4.801832, 5.050528, 4.992812]

# increasing N
Static  = [4.936876, 6.424691, 8.338397, 10.279317]
Dynamic = [4.893106, 6.570677, 8.620886, 10.202264]
Guided  = [4.492695, 6.395298, 8.476779, 10.527868]

#increasing K 
good_matches = 6531
K = [1000,2000,3000,4000,5000,10000]
Static 	= [0.332087, 0.663275, 1.003342, 1.344504, 1.713868, 3.311665]
Dynamic = [0.330207, 0.677518, 1.000835, 1.317273, 1.667087, 3.285113, 17.771486]
Guided 	= [0.343519, 0.674169, 0.993325, 1.360757, 1.643602, ]



#