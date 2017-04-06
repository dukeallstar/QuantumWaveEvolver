import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

#Initialization of lists to hold functions
CurrentWaveFunction = []
NextWaveFunction = []
Potential = []

#Physical Constants
FeedbackPotentialAmplitude = 0
FeedbackPotentialIncriment = .1
ParticleMass = 1
Hbar = 1
mu = 1
WellWidth = 1
WellDepth = 2000
Time = 10*10**(-2)
InitialWavePosition = .3 #Maxima of initial wave function between -1 and 1 which represent the ends of the well

#Simulation Details
Resolution = 100 #Increased resolution (points in the well) requires increased time steps
TimeInterval = 1*10**(-6) #1e-6 is sufficient for resolution of 100 
NSteps = int(Time/TimeInterval)
SpaceInterval = WellWidth/Resolution #Size of the quanta of space
InitialPotentialWellType = 'Box'
InitialWavefunctionType = 'NormalizedGaussian'

#Render Details
RenderFrame = 50 #How many iterations between rendered Frames (100 is good for 1e-6 time steps
RenderTime = 20 #Time to wait between frames if rendering fast enough
TotalFrames = int(Time/(TimeInterval*RenderFrame))
Args = 0 #Place Holder
Record = 0 # 0 = output to screen, 1 = record

# Set up formatting for the movie files
if Record == 1:
	RenderTime = 1
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#Configure the Plot
fig1 = plt.figure()
line, = plt.plot([], [], 'r-')
plt.xlim(-WellWidth, WellWidth)
plt.ylim(0, 15/WellWidth)
plt.xlabel('X')
plt.title('Probability Distribution')

#Modify the potential gradual, either increasing or decresing
def GenerateNextFeedbackPotential(FeedbackState, Type):
	global FeedbackPotentialAmplitude

	if FeedbackState == 1:
		FeedbackPotentialAmplitude += FeedbackPotentialIncriment
	if FeedbackState == 0:
		FeedbackPotentialAmplitude -= FeedbackPotentialIncriment
	
	Potential = []
	
	if Type == 'NormalizedGaussian':
		Box = GeneratePotential(InitialPotentialWellType, WellWidth, WellDepth, Resolution)
		Probe = GenerateInitialWavefunction(WellWidth, Resolution, 'NormalizedGaussian', (InitialWavePosition*WellWidth))
		for Index in range(len(Box)):
			Potential.append((Box[Index][0], Box[Index][1]+FeedbackPotentialAmplitude*Probe[Index][1]))
	return Potential
			

#Evaluate Wavefunctions total energy and return it
def CalculateEnergy (WaveFunction, Potential):

	WaveFunctionDerivative = FunctionDerivative(WaveFunction)
	WaveFunctionSecondDerivative = FunctionDerivative(WaveFunctionDerivative)
	Energy = 0+0j
	
	for Index in range(len(WaveFunction)):
		D2Y = WaveFunctionSecondDerivative[Index][1]
		HPsi = -((Hbar**2)/(2*mu))*D2Y + (Potential[Index][1]*WaveFunction[Index][1])
		PsiHPsi = np.conj(WaveFunction[Index][1])*HPsi
		Energy += PsiHPsi*SpaceInterval
		
	return Energy

#Generic non-normalized Gaussian
def Gaussian (Mean, StandardDeviation, X):
	Normalization = (1/np.sqrt(2*np.pi*StandardDeviation**2))
	Amplitude = np.exp(-((X-Mean)**2)/(2*StandardDeviation**2))
	return (Normalization * Amplitude)

#Gaussian curve normalized for probability distribution to total 1
def NormalizedProbabilityGaussian (Mean, StandardDeviation, X):
	Normalization = (1/np.sqrt(np.sqrt(np.pi*StandardDeviation**2)))
	Amplitude = np.exp(-((X-Mean)**2)/(2*StandardDeviation**2))
	return (Normalization * Amplitude)

#Generates an initial wave function in the form of a list of numbers with an x and y
def GenerateInitialWavefunction (WellWidth, Resolution, WaveType, WaveCenter):
	WaveFunction = []
	if WaveType == 'Cos':
		for Interval in range(-100, 0):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			WaveFunction.append((x,0))
		for Interval in range(0,Resolution+1):
			x = (Interval*WellWidth/Resolution) - (WellWidth/2)
			WaveFunction.append((x,np.cos(3.14*(x/WellWidth))))
		for Interval in range(Resolution+1, Resolution+101):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			WaveFunction.append((x,0))
		return WaveFunction
	
	if WaveType == 'Gaussian':
		for Interval in range(-100, 0):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			WaveFunction.append((x,0))
		for Interval in range(0,Resolution+1):
			x = (Interval*WellWidth/Resolution) - (WellWidth/2)
			WaveFunction.append((x, Gaussian(WaveCenter, WellWidth/20, x)))
		for Interval in range(Resolution+1, Resolution+101):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			WaveFunction.append((x,0))
		return WaveFunction
	
	if WaveType == 'NormalizedGaussian':
		for Interval in range(-100, 0):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			WaveFunction.append((x,0))
		for Interval in range(0,Resolution+1):
			x = (Interval*WellWidth/Resolution) - (WellWidth/2)
			WaveFunction.append((x, NormalizedProbabilityGaussian(WaveCenter, WellWidth/20, x)))
		for Interval in range(Resolution+1, Resolution+101):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			WaveFunction.append((x,0))
		return WaveFunction

#Generates a potential in the form of a list of numbers with x and y
def GeneratePotential (WellType, WellWidth, WellDepth, Resolution):
	Potential = []
	if WellType == 'Box':
		for Interval in range(-100, 0):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,WellDepth))
		for Interval in range(0,Resolution+1):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,0))
		for Interval in range(Resolution+1, Resolution+101):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,WellDepth))
		return Potential
	
	if WellType == 'Box+Bar': #A potential with a square barrier in the middle
		for Interval in range(-100, 0):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,WellDepth))
		for Interval in range(0,(int(Resolution/2)-int(Resolution*0.1))):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,0))
		for Interval in range((int(Resolution/2)-int(Resolution*0.1)),(int(Resolution/2)+int(Resolution*0.1))):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,200))
		for Interval in range((int(Resolution/2)+int(Resolution*0.1)),Resolution+1):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,0))
		for Interval in range(Resolution+1, Resolution+101):
			x = Interval*WellWidth/Resolution - (WellWidth/2)
			Potential.append((x,WellDepth))
		return Potential

#Draw a single plot to the Figure
def DrawPlot (Function):
	X = []
	Y = []
	for point in Function:
		X.append(point[0])
		Y.append(point[1])
	line = plt.plot(X, Y)
	
	return line

#Find the derivative of a function provided as a list of values using nearest neighbour
def FunctionDerivative (Function):
	FunctionDerivative = []
	for Index in range(len(Function)):
		if Index == 0:
			XPrevious = Function[Index][0]
		else:
			XPrevious = Function[Index-1][0]
		if Index == (len(Function)-1):
			XNext = Function[Index][0]
		else:
			XNext = Function[Index+1][0]
		if Index == 0:	
			YPrevious = Function[Index][1]
		else:
			YPrevious = Function[Index-1][1]
		if Index == (len(Function)-1):
			YNext = Function[Index][1]
		else:
			YNext = Function[Index+1][1]
			
		Derivative = (YNext-YPrevious)/(XNext-XPrevious)
		FunctionDerivative.append((Function[Index][0], Derivative))
	return FunctionDerivative

#Calculates the probability distribution based on the WaveFunction
def CalculateProbability (WaveFunction):
	Probability = []
	for point in WaveFunction:
		Probability.append((point[0], point[1].real**2 + point[1].imag**2))
	return Probability

#Iterate the wave function in time (currently only RK = 1 or Euler aproximation)
def CalculateNextWaveFunction (WaveFunction, Potential, TimeStep, RKOrder):
	NextWaveFunction = []
	WaveFunctionDerivative = FunctionDerivative(WaveFunction)
	WaveFunctionSecondDerivative = FunctionDerivative(WaveFunctionDerivative)
	TimeDerivative = 0
	if RKOrder == 1:
		for Index in range(len(WaveFunction)):
			D2Y = WaveFunctionSecondDerivative[Index][1]
			KineticTerm = ((1j*Hbar)/(2*ParticleMass))* D2Y
			PotentialTerm = Potential[Index][1]*(-1j/Hbar)*WaveFunction[Index][1]
			TimeDerivative = KineticTerm + PotentialTerm
			X = WaveFunction[Index][0]
			Y = WaveFunction[Index][1]
			NewY = Y + TimeDerivative * TimeStep
			
			if Index >= (len(WaveFunction)-10):
				NewY=0
			if Index <= 10:
				NewY=0
				
			NextWaveFunction.append((X, NewY))
		return NextWaveFunction

#Calculates the total probability to find the particle as a sanity check (should = 1)
def CalculateTotalProbability(ProbabilityDistribution, SpaceInterval):
	TotalProbability = 0
	for Point in ProbabilityDistribution:
		TotalProbability += Point[1]*SpaceInterval
	return TotalProbability

#Runs each frame to update the line to be plotted and any other main loop functions
def Update_Line(Frame, Lines, Args): 
	global ProbabilityDistribution
	global CurrentWaveFunction
	global Potential
	
	print(Frame)
	print(CalculateTotalProbability(ProbabilityDistribution, SpaceInterval))
	print(CalculateEnergy(CurrentWaveFunction, Potential))
	
	X = []
	Y = []
	for point in ProbabilityDistribution:
		X.append(point[0])
		Y.append(point[1])
	Lines[0].set_data(X,Y)
	
	X = []
	Y = []
	for point in Potential:
		X.append(point[0])
		Y.append(point[1])
	Lines[1].set_data(X,Y)
	
	for i in range(RenderFrame):
		CurrentWaveFunction = CalculateNextWaveFunction(CurrentWaveFunction, Potential, TimeInterval,1)
		ProbabilityDistribution = CalculateProbability(CurrentWaveFunction)
	
	Potential = GenerateNextFeedbackPotential(1, 'NormalizedGaussian')
	
	return Lines

#Setup initial plot
def SetupPlot(ProbabilityFunction, Potential):
	plt.figure(1)
	Ax1 = plt.subplot(211)
	Line1, = DrawPlot(ProbabilityFunction)
	
	Ax2 = plt.subplot(212)
	Line2, = DrawPlot(Potential)
	Lines = [Line1, Line2]
	return Lines
	
#Generate initial conditions	
CurrentWaveFunction = GenerateInitialWavefunction(WellWidth, Resolution, InitialWavefunctionType, (InitialWavePosition*WellWidth))
Potential = GeneratePotential(InitialPotentialWellType, WellWidth, WellDepth, Resolution)
ProbabilityDistribution = CalculateProbability(CurrentWaveFunction)
Lines = SetupPlot(ProbabilityDistribution, Potential)

#Create animation instance
line_ani = FuncAnimation(fig1, Update_Line, TotalFrames, fargs=(Lines, Args), interval=RenderTime, blit=True)


#Begin animation or recording
if Record == 1:
	line_ani.save('lines3.mp4', writer=writer)
	#line_ani.save('lines.mp4', writer=None, fps=None, dpi=None, codec=None, bitrate=None, extra_args=None, metadata=None, extra_anim=None, savefig_kwargs=None)
else:
	plt.show()
