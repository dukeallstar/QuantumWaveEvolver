
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



#spacial resolution of the simulation
res = .5


# defines the time internval for evolving the system
deltaT =0.5*res**2

#lambda  = 2*epislon**2/deltaT
lam = 2*res*res/deltaT

#length of the well
wellWidth = 100
#height of well, currently it is just infinite since the wavefunction is artificially zeroed at either side of the well
wellHeight = 10000

#number of points used in modeling the wavefunction
numPoints = int(np.ceil(wellWidth/res))

coef = np.zeros((3,numPoints-2),dtype=np.complex_)

coef[1,0] = 1+2.j/lam
coef[2,0] = -1.j/lam

coef[1,numPoints-3] = 1+2.j/lam
coef[0,numPoints-3] = -1.j/lam

for i in range(1,numPoints-3):
	coef[0,i] = -1.j/lam
	coef[1,i] = 1+2.j/lam
	coef[2,i] = -1.j/lam


#Configure the Plot
fig1 = plt.figure()
line, = plt.plot([], [], 'r-')
plt.xlabel('X')
plt.title('Probability Distribution')

Args = 0 #Place Holder

#Gaussian curve normalized for probability distribution to total 1?
def normGaus2D (point,center,sigma):
	sigmaX = sigma[0]
	sigmaY = sigma[1]
	x = point[0]
	y = point[1]
	centerX = center[0]
	centerY = center[1]

	value = 1.0/(2*np.pi*sigmaX*sigmaY)*np.exp(-(((x-centerX)**2/(2*sigmaX**2))+((y-centerY)**2/(2*sigmaY**2))))*np.exp(1.j*-61.0*(x+y))
	return value

#Generates an 2d guassian wave
def initWave (center,sigma):
	waveFunction = np.zeros((numPoints,numPoints),dtype=complex)


	for i in range(1,numPoints-1):
		for j in range(1,numPoints-1):
			point = np.array([i*res,j*res])
			waveFunction[i,j] = normGaus2D(point,center,sigma)

	return waveFunction


def potential():
	wellShape = np.zeros((numPoints,numPoints),dtype=np.complex_)

	for i in range(int(numPoints/2.0),int(numPoints/2.0)+10):
		for j in range(int(numPoints/2.0),int(numPoints/2.0)+10):
			wellShape[i][j]=5.0
	return wellShape

def potentialToGraph():
	wellShape = np.zeros((numPoints,numPoints),dtype=np.complex_)

	for i in range(int(numPoints/2.0),int(numPoints/2.0)+10):
		for j in range(int(numPoints/2.0),int(numPoints/2.0)+10):
			wellShape[i][j]=.01
	return wellShape


def solveTriDiag(a,b,c,u,r,m):
	gam = np.zeros(numPoints-2,dtype=complex)
	bet=b[0]
	u[1,m] = r[1,m]/bet

	for j in range(1,numPoints-2):
		gam[j]=c[j-1]/bet;
		bet = b[j]-a[j]*gam[j]
		u[j+1,m] = (r[j+1,m]-a[j]*u[j,m])/bet

	for j in range(numPoints-4,-1,-1):
		u[j+1,m] -= gam[j+1]*u[j+2,m]

	return u

def solveTriDiag2(a,b,c,u,r,m):
	gam = np.zeros(numPoints-2,dtype=complex)
	bet=b[0]
	u[m,1] = r[m,1]/bet

	for j in range(1,numPoints-2):
		gam[j]=c[j-1]/bet;
		bet = b[j]-a[j]*gam[j]
		u[m,j+1] = (r[m,j+1]-a[j]*u[m,j])/bet

	for j in range(numPoints-4,-1,-1):
		u[m,j+1] -= gam[j+1]*u[m,j+2]

	return u

def timeStep(psi,V):

	#left-hand side of 1st intermediate equation
	chi = np.zeros((numPoints,numPoints),dtype=complex)
	psiStar =np.zeros((numPoints,numPoints),dtype=complex)



	for i in range(1,numPoints-1):
		for j in range(1,numPoints-1):
			chi[i,j] = psi[i,j] + (1.j/lam)*(psi[i,j+1]-2*psi[i,j]+psi[i,j-1])


	for m in range(1,numPoints-2):
		psiStar = solveTriDiag(coef[0],coef[1],coef[2],psiStar,chi,m)


	phiStar = np.zeros((numPoints,numPoints),dtype=complex)

	for i in range(0,numPoints):
		for j in range(0,numPoints):
			phiStar[i,j] = np.exp(-1.j*deltaT*V[i,j]/2.0)*psiStar[i,j]

	eta = np.zeros((numPoints,numPoints),dtype=complex)

	for i in range(1,numPoints-1):
		for j in range(1,numPoints-1):
			eta[i,j] = phiStar[i,j] + (1.j/lam)*(phiStar[i+1,j]-2*phiStar[i,j]+phiStar[i-1,j])


	phi=np.zeros((numPoints,numPoints),dtype=complex)

	for m in range(1,numPoints-2):
		phi = solveTriDiag2(coef[0],coef[1],coef[2],phi,eta,m)

	nextPsi = np.zeros((numPoints,numPoints),dtype=complex)

	for i in range(0,numPoints):
		for j in range(0,numPoints):
			nextPsi[i,j] = phi[i,j]/np.exp(1.j*deltaT*V[i,j]/2.0)


	return nextPsi


#Calculates the probability distribution based on the WaveFunction
def prob (psi):

	return np.absolute(psi)


def Update(Frame):
	global psi
	global psiSquared
	global V
	global surface
	global X
	global Y

	psi = timeStep(psi,V)
	psiSquared = prob(psi)


	ax.collections.remove(surface)
	surface = ax.plot_surface(X, Y, psiSquared, cmap=cm.coolwarm)
	print(Frame)



fig = plt.figure(1)
ax = fig.add_subplot(111, projection = '3d')
x = []
y=[]
for i in range(0,numPoints):
	x.append(i*res)
	y.append(i*res)
X=[]
Y=[]
X,Y=np.meshgrid(x,y)

psi = initWave([25,25],[3,3])
V = potential()
Vg = potentialToGraph()
psiSquared = prob(psi)

surface = ax.plot_surface(X, Y, psiSquared, cmap=cm.coolwarm)
surface2 = ax.plot_surface(X,Y,Vg,cmap=cm.coolwarm)



AnimationReference = animation.FuncAnimation(fig, Update, 100)

plt.show()


#ProbabilityDistribution = CalculateProbability(currentWaveFunction)
#print(CalculateTotalProbability(ProbabilityDistribution))


#plt.plot(X,CalculateProbability(currentWaveFunction))
