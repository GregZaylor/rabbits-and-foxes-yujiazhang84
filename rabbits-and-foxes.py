# Rabbits and foxes
# 
# There are initially 400 rabbits and 200 foxes on a farm (but it could be two cell types
# in a 96 well plate or something, if you prefer bio-engineering analogies).
# Plot the concentration of foxes and rabbits as a function of time for a period
# of up to 600 days. The predator-prey relationships are given by the following set of
# coupled ordinary differential equations:
#
# dR/dt = k_1*R - k_2*R*F
# dF/dt = k_3*R*F - k_4*F
#
#
# Constant for growth of rabbits k_1 = 0.015 day^-1
# Constant for death of rabbits being eaten by foxes k_2 = 0.00004 day^-1foxes^-1
# Constant for growth of foxes after eating rabbits k_3 = 0.0004 day^-1rabbits^-1
# Constant for death of foxes k_1 = 0.04 day^-1
# 
# Also plot the number of foxes versus the number of rabbits.
# 
# Then try also with
# k_3 = 0.00004 day^-1 rabbits^-1
# t_final = 800 days
# 
# This problem is based on one from Chapter 1 of H. Scott Fogler's textbook
# "Essentials of Chemical Reaction Engineering".


#Class practice Euler's method

import numpy as np
from matplotlib import pyplot as plt

k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
end_day = 600
step_size = 0.5
R = np.arange(0, end_day, step_size)
F = np.arange(0, end_day, step_size)
R[0] = 400
F[0] = 200
i = len(R)
delta_t = end_day / i
time = np.arange(0, end_day, step_size)

for n in range(i - 1):
    R[n + 1] = R[n] + delta_t * (k1 * R[n] - k2 * R[n] * F[n])
    F[n + 1] = F[n] + delta_t * (k3 * R[n] * F[n] - k4 * F[n])

plt.plot(time, R)
plt.plot(time, F)


#Assignment: odeint function
from scipy.integrate import odeint


# Kinetic constant
k_1 = 0.015
k_2 = 0.00004
k_3 = 0.0004
k_4 = 0.04

# Initial condition
rabbit_0 = 400
fox_0 = 200

# Define
day = 600
step_size = 0.1


# Define function
def f(y, t):
    R, F = y
    dR = k_1 * R - k_2 * R * F
    dF = k_3 * R * F - k_4 * F
    return (dR, dF)


# Solve ODE
initial = [rabbit_0, fox_0]
times = np.arange(0, day, step_size)
result = odeint(f, initial, times)

# Result
rabbit = result[:, 0]
fox = result[:, 1]

# Plot
plt.plot(times, rabbit, label='Rabbits')
plt.plot(times, fox, label='Fox')
plt.xlabel('Day')
plt.ylabel('Number of animal')
plt.legend(loc='upper center')
plt.show()

# Second peak and corresponding day
fox_secondhalf = fox[len(fox) // 2:]
times_secondhalf = times[len(times) // 2:]
peak_fox = int(np.round(np.max(fox_secondhalf)))
peak_times = int(np.round(times_secondhalf[fox_secondhalf.argmax()]))

print("The second peak of fox population is", peak_fox, "at", peak_times, "day.")

#Assignment: KMC

# get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt

# Kintetic constant
k__1 = 0.015
k__2 = 0.00004
k__3 = 0.0004
k__4 = 0.04

# Initial condition
rabbits_0 = 400
foxes_0 = 200

# Define
j = 0
count_fox_dieout = 0
count_rabbit_dieout = 0
fox_peaks = []
corresponding_ts = []
running_print = 1

while j < 100:

    # Create empty list and initiate
    rabbits = []
    foxes = []
    t = []
    rabbits.append(rabbits_0)
    foxes.append(foxes_0)
    t.append(0)

    for i in range(50000):

        # Calculate KMC parameter
        r_k1 = k__1 * rabbits[i]
        r_k2 = k__2 * rabbits[i] * foxes[i]
        r_k3 = k__3 * rabbits[i] * foxes[i]
        r_k4 = k__4 * foxes[i]

        R_k1 = r_k1
        R_k2 = r_k1 + r_k2
        R_k3 = r_k1 + r_k2 + r_k3
        Q_k = r_k1 + r_k2 + r_k3 + r_k4

        # Count the situation rabbits and foxes both die out (Q_k = 0)
        if Q_k == 0.0:
            #            count_rabbit_dieout = count_rabbit_dieout + 1
            j += 1
            break

        u = 1.0 - np.random.rand()
        uQ = u * Q_k

        # Decide which event happens
        if uQ <= R_k1:
            rabbits.append(rabbits[i] + 1)
            foxes.append(foxes[i])
        elif R_k1 < uQ <= R_k2:
            rabbits.append(rabbits[i] - 1)
            foxes.append(foxes[i])
        elif R_k2 < uQ <= R_k3:
            rabbits.append(rabbits[i])
            foxes.append(foxes[i] + 1)
        elif R_k3 < uQ <= Q_k:
            rabbits.append(rabbits[i])
            foxes.append(foxes[i] - 1)

        # Update the time
        v = 1.0 - np.random.rand()
        delta_t = (1 / Q_k) * (np.log(1 / v))
        t.append(t[i] + delta_t)

        # Set the end day
        if t[-1] >= 600:
            j += 1
            break

        # Count the situation foxes die out
    if foxes[-1] == 0:
        count_fox_dieout = count_fox_dieout + 1

    # Plot the situation foxes don't die out
    elif rabbits[-1] != 0:
        plt.plot(t, rabbits, 'b')
        plt.plot(t, foxes, 'g')

        # Second peaks
        foxes_secondhalf = foxes[len(foxes) // 2:]
        t_secondhalf = t[len(t) // 2:]
        peak_foxes = int(round(np.max(foxes_secondhalf)))
        peak_t = int(np.round(t_secondhalf[foxes_secondhalf.index(np.max(foxes_secondhalf))]))
        fox_peaks.append(peak_foxes)
        corresponding_ts.append(peak_t)

    # Just want to know whether the program is still running
    if j == 100 * running_print:
        print(j, "trials have been done.")
        running_print += 1

# Result calculation
# 1. Expected second peak
ave_peak = int(round(np.average(fox_peaks)))
ave_t = int(round(np.average(corresponding_ts)))

# 2. Interquartile range 
IQR1_peak = int(np.round(np.percentile(fox_peaks, [25])))
IQR3_peak = int(np.round(np.percentile(fox_peaks, [75])))
IQR1_t = int(np.round(np.percentile(corresponding_ts, [25])))
IQR3_t = int(np.round(np.percentile(corresponding_ts, [75])))

# 3. Possibility
possibility = np.round(count_fox_dieout / j * 100, 2)

# Plot
plt.legend(['Rabbit', 'Fox'])
plt.xlabel('Day')
plt.ylabel('Population')
plt.show()

# Answer
print("The results are based on", j, "trials:")
print("1. The expected location of the second peak in foxes:", ave_peak, "foxes at", ave_t, "day.")
print("2. The interquartile range of the second peak in foxes:", IQR1_peak, "-", IQR3_peak, "foxes at",
      IQR1_t, "-", IQR3_t, "days.")
print("3. The probability that the foxes die out before 600 days are complete:", possibility, "%.")

#Things learned from this assignment
print("Things learned from this assignment:")
print(
    "1. Different types of containers have different operators. For example, array doesn't have name.index, and list doesn't have name.argmax. Choosing the right type of container coulde make work easier. However, sometimes we have to choose one type instead of the other, just like it has to be list for the KMC.")
print(
    "2. Making use of other resources is very helpful. Every time having a problem that needs a long code to carry out, Google it first, and we may find there is a simple operation from a library. Since we are all using the GitHub, code written by our classmates is a good reference (but we need to learn it, not simply copying it).")
print("3. Understanding error messages is very important.")
