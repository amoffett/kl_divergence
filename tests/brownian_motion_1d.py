import numpy as np
import matplotlib.pyplot as plt

class brownian_dynamics_1d():
    
    def __init__(self, n_particles = 1, gamma = 1, T = 1, bound = [0, 360], V = np.array([[-.5, -.001, 90], \
                                                                                          [-1, -.001, 180], \
                                                                                          [-.2, -.001, 270]])):
        self.k_B = 1
        self.n_particles = 1
        self.gamma = gamma
        self.T = T
        self.bound = bound
        self.V = V
    
    def potential(self, X):
        value = 0
        for i in range(self.V.shape[0]):
            value += self.V[i,0] * np.exp(self.V[i,1] * (X - self.V[i,2])**2)
        return value
    
    def force(self, X, k_bound = 1):
        value = 0
        for i in range(self.V.shape[0]):
            value += -1 * (2 * self.V[i,0] * self.V[i,1] * (X - self.V[i,2]) * np.exp(self.V[i,1] * (X - self.V[i,2])**2))
        if X < self.bound[0]:
            value += abs(k_bound * (X + self.bound[0]))
        elif X > self.bound[1]:
            value += -1 * abs(k_bound * (X - self.bound[1]))
        return value
                           
    def random_force(self, dt = 1):
        mean = 0
        var = np.sqrt(2 * (dt * self.k_B * self.T) / self.gamma)
        value = np.random.normal(mean, var)
        return value

    def plot_potential(self):
        minx = self.bound[0]
        maxx = self.bound[1]
        grid_width = (maxx - minx) / 360.0
        x = np.mgrid[minx:maxx:grid_width]
        V = self.potential(x)
        plt.plot(x, V)
        plt.xlim([minx,maxx])
        plt.xlabel(r'X')
        plt.ylabel(r'V(X)')
        
    def plot_probability(self, hist = False):
        try:
            self.X_t
        except AttributeError:
            print "Run a simulation first."
            exit(2)
        minx = self.bound[0]
        maxx = self.bound[1]
        grid_width = (maxx - minx) / 360.0
        x = np.mgrid[minx:maxx:grid_width]
        V = self.potential(x)
        p = np.exp(-V/(self.k_B*self.T))/np.sum(np.exp(-V/(self.k_B*self.T)))
        if hist == True:
            plt.hist(self.X_t, range = [minx, maxx], bins = maxx - minx - 1, normed = True)
        plt.plot(x, p)
        plt.xlim([minx,maxx])
        plt.xlabel(r'X')
        plt.ylabel(r'P(X)')
        
    def integrate_brownian(self, X, n_steps, dt):
        step = 0
        self.X_t = [X]
        self.E_t = [self.potential(X)]
        while step < n_steps:
            X = self.X_t[step] + self.force(self.X_t[step]) / self.gamma * dt + self.random_force(dt = dt)
            self.X_t.append(X)
            self.E_t.append(self.potential(X))
            step += 1
        self.X_t = np.vstack(self.X_t)
        self.E_t = np.vstack(self.E_t)
