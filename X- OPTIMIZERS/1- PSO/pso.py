class particle:
	def __init__ (self,position):
		self.position = position
		self.bestPositionKnown = []
		self.bestParticleKown = []

class particle_swarm_optimization:
	def run(func,particlesNumb,maxInterations):
