+++ 
draft = true
date = 2020-01-05T20:48:43+01:00
title = "Swarm Particle Interpolation"
description = ""
slug = "swarm-particle-interpolation" 
tags = []
categories = []
externalLink = ""
series = []
+++


#### Adapted to the N dimensional case
***

Visit the github repository for the full code.
[![GitHub](/icons/github-icon.png)](https://github.com/viclule/particle_swarm_optimization_py)
<br/>
<br/>
It is a professional implementation based on the [article from Iran Macedo.](https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6)
<br/>
<br/>
#### Theory

[![Test for 3D](/images/posts/swarm_particle_optimization.png)]
<br/>
<br/>

"In computational science, particle swarm optimization (PSO) [1] is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search-space according to simple mathematical formulae over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions." Wikipedia
***
#### Python code

```python
import random
import numpy as np 


class Particle():
    """
    A particle of the swarm
    """
    def __init__(self, dimensions):
        """
        Init
            :param self: 
            :param dimensions: number of dimensions for the particle
        """
        self._initialize_position(dimensions)
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.zeros(dimensions)

    def _initialize_position(self, dimensions):
        """
        Initialize to a random position
            :param self: 
            :param dimensions: number of dimensions to optimize
        """
        self.position = np.zeros(dimensions)
        for d in range(dimensions):
            self.position[d] = (-1)**(bool(random.getrandbits(1))) * \
                                random.random()*50

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)
    
    def move(self):
        """
        Update the particle position
            :param self: 
        """
        self.position = self.position + self.velocity


# jumping some lines...

```