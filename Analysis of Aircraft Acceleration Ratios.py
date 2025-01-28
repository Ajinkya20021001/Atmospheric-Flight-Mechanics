import math
import matplotlib.pyplot as plt

g0 = 9.81  # (m/s^2)
Re = 6378000  # (m)
Ve = 463.82  # (m/s)
w = ((2 * math.pi) / (24 * 3600) )

Vr = 0.0
h = 0.0
Vac = 0.0

j = 1
while j <= 10:  
    Vac = 100 * j
    Vr = (Re * w) + Vac

    h_values = []  
    ac_g_values = []  
    i = 1
    while i <= 120:
        h = 500 * i  
        g = g0 * ((Re / (Re + h)) ** 2)  
        ac = (Vr ** 2) / (Re + h)  
        ac_g = (ac / g)  
        
        
        h_values.append(h)
        ac_g_values.append(ac_g)
        
        i += 1
    
    
    plt.plot(h_values, ac_g_values, label=f'Vac = {Vac} m/s')

    j += 1


plt.xlabel('Height (h) in meters')
plt.ylabel('(ac/g) (Acceleration/Gravity)')
plt.title('ac/g vs Height (h) for Different Vac Values')
plt.grid(True)


plt.legend()


plt.show()
