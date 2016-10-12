
# coding: utf-8

# # Rabbits and foxes
# 
# There are initially 400 rabbits and 200 foxes on a farm (but it could be two cell types in a 96 well plate or something, if you prefer bio-engineering analogies). Plot the concentration of foxes and rabbits as a function of time for a period of up to 600 days. The predator-prey relationships are given by the following set of coupled ordinary differential equations:
# 
# \begin{align}
# \frac{dR}{dt} &= k_1 R - k_2 R F \tag{1}\\
# \frac{dF}{dt} &= k_3 R F - k_4 F \tag{2}\\
# \end{align}
# 
# * Constant for growth of rabbits $k_1 = 0.015$ day<sup>-1</sup>
# * Constant for death of rabbits being eaten by foxes $k_2 = 0.00004$ day<sup>-1</sup> foxes<sup>-1</sup>
# * Constant for growth of foxes after eating rabbits $k_3 = 0.0004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * Constant for death of foxes $k_1 = 0.04$ day<sup>-1</sup>
# 
# Also plot the number of foxes versus the number of rabbits.
# 
# Then try also with 
# * $k_3 = 0.00004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * $t_{final} = 800$ days
# 
# *This problem is based on one from Chapter 1 of H. Scott Fogler's textbook "Essentials of Chemical Reaction Engineering".*
# 

# # Solving ODEs
# 
# *Much of the following content reused under Creative Commons Attribution license CC-BY 4.0, code under MIT license (c)2014 L.A. Barba, G.F. Forsyth. Partly based on David Ketcheson's pendulum lesson, also under CC-BY. https://github.com/numerical-mooc/numerical-mooc*
# 
# Let's step back for a moment. Suppose we have a first-order ODE $u'=f(u)$. You know that if we were to integrate this, there would be an arbitrary constant of integration. To find its value, we do need to know one point on the curve $(t, u)$. When the derivative in the ODE is with respect to time, we call that point the _initial value_ and write something like this:
# 
# $$u(t=0)=u_0$$
# 
# In the case of a second-order ODE, we already saw how to write it as a system of first-order ODEs, and we would need an initial value for each equation: two conditions are needed to determine our constants of integration. The same applies for higher-order ODEs: if it is of order $n$, we can write it as $n$ first-order equations, and we need $n$ known values. If we have that data, we call the problem an _initial value problem_.
# 
# Remember the definition of a derivative? The derivative represents the slope of the tangent at a point of the curve $u=u(t)$, and the definition of the derivative $u'$ for a function is:
# 
# $$u'(t) = \lim_{\Delta t\rightarrow 0} \frac{u(t+\Delta t)-u(t)}{\Delta t}$$
# 
# If the step $\Delta t$ is already very small, we can _approximate_ the derivative by dropping the limit. We can write:
# 
# $$\begin{equation}
# u(t+\Delta t) \approx u(t) + u'(t) \Delta t
# \end{equation}$$
# 
# With this equation, and because we know $u'(t)=f(u)$, if we have an initial value, we can step by $\Delta t$ and find the value of $u(t+\Delta t)$, then we can take this value, and find $u(t+2\Delta t)$, and so on: we say that we _step in time_, numerically finding the solution $u(t)$ for a range of values: $t_1, t_2, t_3 \cdots$, each separated by $\Delta t$. The numerical solution of the ODE is simply the table of values $t_i, u_i$ that results from this process.
# 

# # Euler's method
# *Also known as "Simple Euler" or sometimes "Simple Error".*
# 
# The approximate solution at time $t_n$ is $u_n$, and the numerical solution of the differential equation consists of computing a sequence of approximate solutions by the following formula, based on Equation (10):
# 
# $$u_{n+1} = u_n + \Delta t \,f(u_n).$$
# 
# This formula is called **Euler's method**.
# 
# For the equations of the rabbits and foxes, Euler's method gives the following algorithm that we need to implement in code:
# 
# \begin{align}
# R_{n+1} & = R_n + \Delta t \left(k_1 R_n - k_2 R_n F_n \right) \\
# F_{n+1} & = F_n + \Delta t \left( k_3 R_n F-n - k_4 F_n \right).
# \end{align}
# 

# In[1]:

### Class practice: Euler's method


# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt


# In[3]:

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

for n in range(i-1):
    R[n+1] = R[n] + delta_t * (k1*R[n] - k2*R[n]*F[n])
    F[n+1] = F[n] + delta_t * (k3*R[n]*F[n] - k4*F[n])
    
plt.plot(time, R)
plt.plot(time, F)


# In[4]:

### Assignment: odeint function


# In[5]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint


# In[6]:

# Kintetic constant
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
    dR = k_1*R - k_2*R*F
    dF = k_3*R*F - k_4*F
    return (dR, dF)

# Solve ODE
initial = [rabbit_0, fox_0]
times = np.arange(0, day, step_size)
result = odeint (f, initial, times)

# Result
rabbit = result[:,0]
fox = result[:,1]

# Plot
plt.plot(times, rabbit, label='Rabbits')
plt.plot(times, fox, label='Fox')
plt.xlabel('Day')
plt.ylabel('Number of animal')
plt.legend(loc='upper center')
plt.show()

# Second peak and corresponding day
fox_secondhalf = fox[len(fox)//2:]
times_secondhalf = times[len(times)//2:]
peak_fox = int(np.round(np.max(fox_secondhalf)))
peak_times = int(np.round(times_secondhalf[fox_secondhalf.argmax()]))

print("The second peak of fox population is", peak_fox, "at", peak_times, "day.")


# In[7]:

### Assignment: KMC


# In[8]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt


# In[9]:

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

while j < 10000:

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
            j = j + 1
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
        delta_t = (1/Q_k) * (np.log(1/v))
        t.append(t[i] + delta_t)

# Set the end day
        if t[-1] >=600:
            j = j + 1
            break

# Count the situation foxes die out
    if foxes[-1] == 0:
        count_fox_dieout = count_fox_dieout + 1

# Plot the situation foxes don't die out
    elif rabbits[-1] != 0:
        plt.plot(t, rabbits, 'b')
        plt.plot(t, foxes, 'g')

# Second peaks 
        foxes_secondhalf = foxes[len(foxes)//2:]
        t_secondhalf = t[len(t)//2:]
        peak_foxes = int(round(np.max(foxes_secondhalf)))
        peak_t = int(np.round(t_secondhalf[foxes_secondhalf.index(np.max(foxes_secondhalf))]))
        fox_peaks.append(peak_foxes)
        corresponding_ts.append(peak_t)

# Just want to know whether the program is still running
    if j == 100 * running_print:
        print(j, "trials have been done.")
        running_print = running_print + 1

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


# In[10]:

# Answer
print("The results are based on", j, "trials:")
print("1. The expected location of the second peak in foxes:", ave_peak, "foxes at", ave_t, "day.")
print("2. The interquartile range of the second peak in foxes:", IQR1_peak, "-", IQR3_peak, "foxes at", 
      IQR1_t, "-", IQR3_t, "days.")
print("3. The probability that the foxes die out before 600 days are complete:", possibility, "%.")


# In[11]:

### Things learned from this assignment
print("Things learned from this assignment:")
print("1. Different types of containers have different operators. For example, array doesn't have name.index, and list doesn't have name.argmax. Choosing the right type of container coulde make work easier. However, sometimes we have to choose one type instead of the other, just like it has to be list for the KMC.")
print("2. Making use of other resources is very helpful. Every time having a problem that needs a long code to carry out, Google it first, and we may find there is a simple operation from a library. Since we are all using the GitHub, code written by our classmates is a good reference (but we need to learn it, not simply copying it).")
print("3. Understanding error messages is very important.")

