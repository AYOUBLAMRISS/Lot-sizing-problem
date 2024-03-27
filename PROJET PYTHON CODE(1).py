import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt

# definition des constantes

P = 550

K_m = 50

D = 300

K_r = 10

H_m = 50

H_r = 30

F = 0.05

B = 33

W = 0.002

A = 0.07

E = 0.05

C = 7

# definition des formules de n, q et TC 


def calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E):
  return np.sqrt((2*E*D*(K_m+F*B)+W*(B**2)*A)*(P*delta_H+2*H_m*D)/(2*E*D*H_m*K_r*(P-D)))



def calculate_q(P, D, H_m, K_m, F, B, W, A, E, n):
  return np.sqrt((8*E**2*P*D*H_m*(K_m+F*B)*(P-D)+4*E*P*D*W*(B**2)*A*H_m*(P-D)))/(2*n*E*H_m*(P-D))


def calculate_TC(K_m, K_r, D, n, q, H_m, P, delta_H, F, B, W, A, E, C, S_i):
  return (K_m+n*K_r)*(D/(n*q))+H_m*(D*q/P+((P-D)*n*q)/(2*P))+delta_H*q/2+F*B*D/(n*q)+(W*B**2*A)/(2*n*q*E)+C*D*(2-A)+S_i*D




# Calcul des valeurs optimales de n et q 

delta_H = H_m - H_r

n_opt = calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E)

q_opt = calculate_q(P, D, H_m, K_m, F, B, W, A, E, n_opt)



# determination des coûts de transport basé sur q 

S_i = 0

if q_opt < 5:
   S_i = 4

elif q_opt < 10 and q_opt>=5:
  S_i = 3.5

elif q_opt < 15 and q_opt>=10:
  S_i = 3.2

else:
  S_i = 3.1



# Calcul du cout total avec n et q optimaux

TC_opt = calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i)


# Print the results


print(f"Optimal n: {n_opt}")

print(f"Optimal q: {q_opt}")

print(f"Unit transportation cost (S_i): {S_i}")

print(f"Total cost with optimal n, q, and S_i: {TC_opt}")




# Tracer du cout total en fonction des couts de configuration en pourcentage

D_values = np.linspace(0,300,100) /3

TC_values = [calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i) for D in D_values]

L = [(i/sum(TC_values))*100 for i in TC_values]

T=list(accumulate(L))

print (D_values)

print(L)

print (TC_values)

print(sum(TC_values))

print (T)

plt.plot(D_values,T)

plt.xlabel('DEMAND rate' )

plt.ylabel('Total cost')

plt.show()



-----------------------------------------------------------------------------------------------
Influence en certains paramètres :
1) Km

import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt


# definition des constantes

P = 550

K_m = 50

D = 300

K_r = 10

H_m = 50

H_r = 30

F = 0.05

B = 33

W = 0.002

A = 0.07

E = 0.05

C = 7


# definition des formules de n, q et TC 

def calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E):
  return np.sqrt((2*E*D*(K_m+F*B)+W*(B**2)*A)*(P*delta_H+2*H_m*D)/(2*E*D*H_m*K_r*(P-D)))


def calculate_q(P, D, H_m, K_m, F, B, W, A, E, n):
  return np.sqrt((8*E**2*P*D*H_m*(K_m+F*B)*(P-D)+4*E*P*D*W*(B**2)*A*H_m*(P-D)))/(2*n*E*H_m*(P-D))


def calculate_TC(K_m, K_r, D, n, q, H_m, P, delta_H, F, B, W, A, E, C, S_i):
  return (K_m+n*K_r)*(D/(n*q))+H_m*(D*q/P+((P-D)*n*q)/(2*P))+delta_H*q/2+F*B*D/(n*q)+(W*B**2*A)/(2*n*q*E)+C*D*(2-A)+S_i*D


# Calcul des valeurs optimales de n et q 

delta_H = H_m - H_r

n_opt = calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E)

q_opt = calculate_q(P, D, H_m, K_m, F, B, W, A, E, n_opt)




# determination des coûts de transport basé sur q 


S_i = 0

if q_opt < 5:
   S_i = 4


elif q_opt < 10 and q_opt>=5:
  S_i = 3.5


elif q_opt < 15 and q_opt>=10:
  S_i = 3.2


else:
  S_i = 3.1


# Calcul du cout total avec n et q optimaux


TC_opt = calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i)


# Print the results



print(f"Optimal n: {n_opt}")

print(f"Optimal q: {q_opt}")

print(f"Unit transportation cost (S_i): {S_i}")

print(f"Total cost with optimal n, q, and S_i: {TC_opt}")


# Tracer du cout total en fonction des couts de configuration 

K_m_values = np.linspace(0,50,100)

TC_values = [calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i) for K_m in K_m_values]

L = [(i/sum(TC_values))*100 for i in TC_values]

T=list(accumulate(L))

print (K_m_values)

print(L)

print (TC_values)

print(sum(TC_values))

print (T)

plt.plot(K_m_values,TC_values)

plt.xlabel('set up cost' )

plt.ylabel('Total cost')

plt.show()
-------------------------------------------------------------------------------------------
Km en pourcentage

import numpy as np

from itertools import accumulate

import matplotlib.pyplot as plt





# definition des constantes




P = 550

K_m = 50

D = 300

K_r = 10

H_m = 50

H_r = 30

F = 0.05

B = 33

W = 0.002

A = 0.07

E = 0.05

C = 7

# definition des formules de n, q et TC 

def calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E):

  return np.sqrt((2*E*D*(K_m+F*B)+W*(B**2)*A)*(P*delta_H+2*H_m*D)/(2*E*D*H_m*K_r*(P-D)))



def calculate_q(P, D, H_m, K_m, F, B, W, A, E, n):

  return np.sqrt((8*E**2*P*D*H_m*(K_m+F*B)*(P-D)+4*E*P*D*W*(B**2)*A*H_m*(P-D)))/(2*n*E*H_m*(P-D))




def calculate_TC(K_m, K_r, D, n, q, H_m, P, delta_H, F, B, W, A, E, C, S_i):

  return (K_m+n*K_r)*(D/(n*q))+H_m*(D*q/P+((P-D)*n*q)/(2*P))+delta_H*q/2+F*B*D/(n*q)+(W*B**2*A)/(2*n*q*E)+C*D*(2-A)+S_i*D




# Calcul des valeurs optimales de n et q 

delta_H = H_m - H_r




n_opt = calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E)

q_opt = calculate_q(P, D, H_m, K_m, F, B, W, A, E, n_opt)




# determination des coûts de transport basé sur q 



S_i = 0




if q_opt < 5:

   S_i = 4




elif q_opt < 10 and q_opt>=5:

  S_i = 3.5




elif q_opt < 15 and q_opt>=10:

  S_i = 3.2




else:

  S_i = 3.1

# Calcul du cout total avec n et q optimaux




TC_opt = calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i)




# Print the results




print(f"Optimal n: {n_opt}")




print(f"Optimal q: {q_opt}")




print(f"Unit transportation cost (S_i): {S_i}")




print(f"Total cost with optimal n, q, and S_i: {TC_opt}")




# Tracer du cout total en fonction des couts de configuration en pourcentage




K_m_values = np.linspace(0,50,100)*2




TC_values = [calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i) for K_m in K_m_values]

L = [(((i-min(TC_values))/min(TC_values))*100)/2 for i in TC_values]

T=list(accumulate(L))




print (K_m_values)

print(L)

print (TC_values)

print(sum(TC_values))

print (T)

plt.plot(K_m_values,L)




plt.xlabel('set up cost' )




plt.ylabel('Total cost')




plt.show()


-------------------------------------------------------------------------------------------
Influence du paramètre 
2) Le nombre de livraison : n(opt)

import numpy as np

from itertools import accumulate

import matplotlib.pyplot as plt





# definition des constantes




P = 550

K_m = 50

D = 300

K_r = 10

H_m = 50

H_r = 30

F = 0.05

B = 33

W = 0.002

A = 0.07

E = 0.05

C = 7

# definition des formules de n, q et TC 

def calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E):

  return np.sqrt((2*E*D*(K_m+F*B)+W*(B**2)*A)*(P*delta_H+2*H_m*D)/(2*E*D*H_m*K_r*(P-D)))



def calculate_q(P, D, H_m, K_m, F, B, W, A, E, n):

  return np.sqrt((8*E**2*P*D*H_m*(K_m+F*B)*(P-D)+4*E*P*D*W*(B**2)*A*H_m*(P-D)))/(2*n*E*H_m*(P-D))




def calculate_TC(K_m, K_r, D, n, q, H_m, P, delta_H, F, B, W, A, E, C, S_i):

  return (K_m+n*K_r)*(D/(n*q))+H_m*(D*q/P+((P-D)*n*q)/(2*P))+delta_H*q/2+F*B*D/(n*q)+(W*B**2*A)/(2*n*q*E)+C*D*(2-A)+S_i*D




# Calcul des valeurs optimales de n et q 

delta_H = H_m - H_r




n_opt = calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E)

q_opt = calculate_q(P, D, H_m, K_m, F, B, W, A, E, n_opt)




# determination des coûts de transport basé sur q 




S_i = 0




if q_opt < 5:

   S_i = 4




elif q_opt < 10 and q_opt>=5:

  S_i = 3.5




elif q_opt < 15 and q_opt>=10:

  S_i = 3.2




else:

  S_i = 3.1

# Calcul du cout total avec n et q optimaux



TC_opt = calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i)




# Print the results




print(f"Optimal n: {n_opt}")




print(f"Optimal q: {q_opt}")




print(f"Unit transportation cost (S_i): {S_i}")




print(f"Total cost with optimal n, q, and S_i: {TC_opt}")




# Tracer du cout total en fonction du nombre de livraisons 




n_values = np.linspace(0,10,10)




TC_values = [calculate_TC(K_m, K_r, D, n, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i) for n in n_values]

L = [(i/sum(TC_values))*100 for i in TC_values]

T=list(accumulate(L))




print (K_m_values)

print(L)

print (TC_values)

print(sum(TC_values))

print (T)

plt.plot(n_values,TC_values)




plt.xlabel('Shipment number' )




plt.ylabel('Total cost')




plt.show()


---------------------------------------------------------------------------------------------
Le nombre de livraisons en pourcentage :

import numpy as np

from itertools import accumulate

import matplotlib.pyplot as plt

# definition des constantes


P = 550

K_m = 50

D = 300

K_r = 10

H_m = 50

H_r = 30

F = 0.05

B = 33

W = 0.002

A = 0.07

E = 0.05

C = 7

# definition des formules de n, q et TC 

def calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E):

  return np.sqrt((2*E*D*(K_m+F*B)+W*(B**2)*A)*(P*delta_H+2*H_m*D)/(2*E*D*H_m*K_r*(P-D)))



def calculate_q(P, D, H_m, K_m, F, B, W, A, E, n):

  return np.sqrt((8*E**2*P*D*H_m*(K_m+F*B)*(P-D)+4*E*P*D*W*(B**2)*A*H_m*(P-D)))/(2*n*E*H_m*(P-D))




def calculate_TC(K_m, K_r, D, n, q, H_m, P, delta_H, F, B, W, A, E, C, S_i):

  return (K_m+n*K_r)*(D/(n*q))+H_m*(D*q/P+((P-D)*n*q)/(2*P))+delta_H*q/2+F*B*D/(n*q)+(W*B**2*A)/(2*n*q*E)+C*D*(2-A)+S_i*D




# Calcul des valeurs optimales de n et q 

delta_H = H_m - H_r




n_opt = calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E)

q_opt = calculate_q(P, D, H_m, K_m, F, B, W, A, E, n_opt)




# determination des coûts de transport basé sur q 




S_i = 0




if q_opt < 5:

   S_i = 4




elif q_opt < 10 and q_opt>=5:

  S_i = 3.5




elif q_opt < 15 and q_opt>=10:

  S_i = 3.2




else:

  S_i = 3.1

# Calcul du cout total avec n et q optimaux




TC_opt = calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i)




# Print the results




print(f"Optimal n: {n_opt}")




print(f"Optimal q: {q_opt}")




print(f"Unit transportation cost (S_i): {S_i}")




print(f"Total cost with optimal n, q, and S_i: {TC_opt}")




# Tracer du cout total en fonction du nombre de livraisons en pourcentage




n_values = np.linspace(0,10,10)




TC_values = [calculate_TC(K_m, K_r, D, n, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i) for n in n_values]

L = [(abs(i-min(TC_values))/min(TC_values))*100 for i in TC_values]

T=list(accumulate(L))




print (K_m_values)

print(L)

print (TC_values)

#print(sum(TC_values))

#print (T)

plt.plot(n_values,L)




plt.xlabel('NUMBER OF SHIPEMENT' )




plt.ylabel('Total cost')




plt.show()

----------------------------------------------------------------------------------------------
3) Influence de la taille du lot économique (qopt) :


import numpy as np

from itertools import accumulate

import matplotlib.pyplot as plt


# definition des constantes

P = 550

K_m = 50

D = 300

K_r = 10

H_m = 50

H_r = 30

F = 0.05

B = 33

W = 0.002

A = 0.07

E = 0.05

C = 7

# definition des formules de n, q et TC 

def calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E):

  return np.sqrt((2*E*D*(K_m+F*B)+W*(B**2)*A)*(P*delta_H+2*H_m*D)/(2*E*D*H_m*K_r*(P-D)))



def calculate_q(P, D, H_m, K_m, F, B, W, A, E, n):

  return np.sqrt((8*E**2*P*D*H_m*(K_m+F*B)*(P-D)+4*E*P*D*W*(B**2)*A*H_m*(P-D)))/(2*n*E*H_m*(P-D))




def calculate_TC(K_m, K_r, D, n, q, H_m, P, delta_H, F, B, W, A, E, C, S_i):

  return (K_m+n*K_r)*(D/(n*q))+H_m*(D*q/P+((P-D)*n*q)/(2*P))+delta_H*q/2+F*B*D/(n*q)+(W*B**2*A)/(2*n*q*E)+C*D*(2-A)+S_i*D




# Calcul des valeurs optimales de n et q 

delta_H = H_m - H_r




n_opt = calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E)

q_opt = calculate_q(P, D, H_m, K_m, F, B, W, A, E, n_opt)




# determination des coûts de transport basé sur q 



S_i = 0




if q_opt < 5:

   S_i = 4




elif q_opt < 10 and q_opt>=5:

  S_i = 3.5




elif q_opt < 15 and q_opt>=10:

  S_i = 3.2




else:

  S_i = 3.1

# Calcul du cout total avec n et q optimaux




TC_opt = calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i)




# Print the results




print(f"Optimal n: {n_opt}")




print(f"Optimal q: {q_opt}")




print(f"Unit transportation cost (S_i): {S_i}")




print(f"Total cost with optimal n, q, and S_i: {TC_opt}")




# Tracer du cout total en fonction de la taille de lot économique




q_values = np.linspace(0,20,10)




TC_values = [calculate_TC(K_m, K_r, D, n_opt, q, H_m, P, delta_H, F, B, W, A, E, C, S_i) for q in q_values]

L = [(i/sum(TC_values))*100 for i in TC_values]

T=list(accumulate(L))




print (K_m_values)

print(L)

print (TC_values)

print(sum(TC_values))

print (T)

plt.plot(q_values,TC_values)




plt.xlabel('shipement lot siz' )




plt.ylabel('Total cost')




plt.show()

-------------------------------------------------------------------------------------

Taille du lot économique en pourcentage

import numpy as np

from itertools import accumulate

import matplotlib.pyplot as plt

# definition des constantes

P = 550

K_m = 50

D = 300

K_r = 10

H_m = 50

H_r = 30

F = 0.05

B = 33

W = 0.002

A = 0.07

E = 0.05

C = 7

# definition des formules de n, q et TC 

def calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E):

  return np.sqrt((2*E*D*(K_m+F*B)+W*(B**2)*A)*(P*delta_H+2*H_m*D)/(2*E*D*H_m*K_r*(P-D)))



def calculate_q(P, D, H_m, K_m, F, B, W, A, E, n):

  return np.sqrt((8*E**2*P*D*H_m*(K_m+F*B)*(P-D)+4*E*P*D*W*(B**2)*A*H_m*(P-D)))/(2*n*E*H_m*(P-D))




def calculate_TC(K_m, K_r, D, n, q, H_m, P, delta_H, F, B, W, A, E, C, S_i):

  return (K_m+n*K_r)*(D/(n*q))+H_m*(D*q/P+((P-D)*n*q)/(2*P))+delta_H*q/2+F*B*D/(n*q)+(W*B**2*A)/(2*n*q*E)+C*D*(2-A)+S_i*D




# Calcul des valeurs optimales de n et q 

delta_H = H_m - H_r




n_opt = calculate_n(P, K_m, F, B, D, delta_H, H_m, K_r, W, A, E)

q_opt = calculate_q(P, D, H_m, K_m, F, B, W, A, E, n_opt)




# determination des coûts de transport basé sur q 




S_i = 0




if q_opt < 5:

   S_i = 4




elif q_opt < 10 and q_opt>=5:

  S_i = 3.5




elif q_opt < 15 and q_opt>=10:

  S_i = 3.2




else:

  S_i = 3.1

# Calcul du cout total avec n et q optimaux


TC_opt = calculate_TC(K_m, K_r, D, n_opt, q_opt, H_m, P, delta_H, F, B, W, A, E, C, S_i)




# Print the results




print(f"Optimal n: {n_opt}")




print(f"Optimal q: {q_opt}")




print(f"Unit transportation cost (S_i): {S_i}")




print(f"Total cost with optimal n, q, and S_i: {TC_opt}")




# Plot TC as a function of demand percentage
# Tracer du cout total en fonction de la taille de lot économique en pourcentage



q_values = np.linspace(0,20,10)




TC_values = [calculate_TC(K_m, K_r, D, n_opt, q, H_m, P, delta_H, F, B, W, A, E, C, S_i) for q in q_values]

L = [((i-min(TC_values))/min(TC_values))*100 for i in TC_values]

T=list(accumulate(L))




print (K_m_values)

print(L)

print (TC_values)

print(sum(TC_values))

print (T)

plt.plot(q_values,L)




plt.xlabel('shipement lot siz' )




plt.ylabel('Total cost')




plt.show()
