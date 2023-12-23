
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
#import sympy as sp
#import pandas as pd
#from sympy import *

#define time and measured radiation intensity
t= np.array([0,1,3,5,7,9])
y= np.array([1,0.891,0.708,0.562,0.447,0.355])

#define the suggested function y=A*exp(lambda * t)
def radInt(t,A,lamda):
    return A*np.exp(lamda*t)

#fit the curve and find values of A and lambda
model = curve_fit(radInt,t,y)

#create new array to store the predicted radiation intensity 
rad = np.empty(len(t))

#fill the array by substituting values ot t using A & lambda values
for i in range(len(t)):
    rad[i]=radInt(t[i],model[0][0],model[0][1])
    
#print A & lambda
print('values of A & lambda: ' ,model)

#Calculate R^2 => the closer to 1 the better the values 
print('R^2: ', r2_score(rad,y))

#Plot given original measured values in green and the values predicted by the model in blue

plt.scatter(t,y,color="green")

plt.title("Scatter Plot")

plt.plot(t,rad,color="blue") 
plt.show()

#old unvalid coding I like to keep for future revision
'''
t=[0,1,3,5,7,9]
gamma=[1,0.891,0.708,0.562,0.447,0.355]
lamda,g,_t = sp.symbols('lamda g _t')
#po=[lamda* t_ for t_ in t] 
#gt=[g*t_ for g,t_ in zip(gamma,t)]
f1=sp.Lambda((g,_t),g*_t*exp(lamda*_t))
sum1=[sp.N(f1(g,_t))for g,_t in zip(gamma,t)]

f2=sp.Lambda((g,_t),g*exp(lamda*_t))
sum2=[sp.N(f2(g,_t)) for g,_t in zip(gamma,t)]

f3=sp.Lambda(_t, exp(2*lamda*_t))
sum3=[sp.N(f3(_t)) for _t in t]

f4=sp.Lambda(_t, _t*exp(2*lamda*_t))
sum4=[sp.N(f4(_t)) for _t in t]

#f=[np.array(sum1)-(np.array(sum2)*np.array(sum4)/np.array(sum3))]
f= [a-(b*d/c) for a,b,c,d in zip(sum1,sum2,sum4,sum3)]
#eqn = Eq(f,0)
print(f)
print('lambda = ',solve(f))


#sympy.init_session()
#x,y = symbols('x y')
#eqn = Eq(8*x**2, 7*x+51)
#eqn
#x = sp.symbols('x')
#exact_value = integrate(exp(-x), (x, 0, 1))
'''
