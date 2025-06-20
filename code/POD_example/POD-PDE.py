"""
POD partial differential equation

This script is an example of the POD method applied to a partial differential equation.
We will aproximate the solution of the partial differential equation using the POD method.

@Author: Miquel Baztán
@Date: 19/06/2025-20/06/2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH = "POD_example/data"

class BURGER:
    """
    The partial differential equation we aim to solve is the following:
    (du/dt) = alpha*(d²u/dx²) - u*(du/dx) where x in (0,1), t in (0,T) with
        u(x,0) = sin(pi*x) for all x in (0,1) (initial condition) and
        u(0,t) = u(1,t) = 0 for all t in (0,T] (boundary condition)

    We will discretize the domain using nx and nt snapshots 
    """
    def __init__(self, alpha, T, nt, nx):
        self.alpha = alpha
        self.T     = T
        self.nt    = nt
        self.nx    = nx

        assert alpha*T*nx**2/nt < 0.5, f"Finite differences does not converge ({alpha*T*nx**2/nt})"

        name = PATH + f"/BURGER_alpha={alpha}_T={T}_nt={nt}_nx={nx}.csv"

        try: self.A = pd.read_csv(name, header=None).to_numpy()
        except FileNotFoundError:
            self.A = self.__snapShots__()
            pd.DataFrame(self.A).to_csv(name, index=False, header=False)
    
    @staticmethod
    def initialCondition(x):
        return np.sin(np.pi*x)

    def __snapShots__(self):
        U = np.zeros((self.nt, self.nx))
        U[0,:] = self.initialCondition(np.linspace(0, 1, self.nx))
        U[0,0], U[0,-1] = 0, 0
        dt = self.T/self.nt
        dx = 1/self.nx

        for t in range(self.nt-1):
            for x in range(1, self.nx-1):
                U[t+1,x] = U[t,x] + dt*(self.alpha/dx**2*(U[t,x+1] - 2*U[t,x] + U[t,x-1]) - U[t,x]*(U[t,x+1] - U[t,x])/dx)

        return U

    @staticmethod
    def solution(x,t):
        return None

class HEAT:
    """
    The partial differential equation we aim to solve is the following:
    (du/dt) = k*(d²u/dx²) where x in (0,1), t in (0,T) with
        u(x,0) = 6*sin(pi*x) for all x in (0,1) (initial condition) and
        u(0,t) = u(1,t) = 0 for all t in (0,T] (boundary condition)

    We will discretize the domain using nx and nt snapshots
    """
    def __init__(self, k, T, nt, nx):
        self.k  = k
        self.T  = T
        self.nt = nt
        self.nx = nx

        assert k*T*nx**2/nt < 0.5, f"Finite differences does not converge ({k*T*nx**2/nt})"

        name = PATH + f"/HEAT_k={k}_T={T}_nt={nt}_nx={nx}.csv"

        try: self.A = pd.read_csv(name, header=None).to_numpy()
        except FileNotFoundError:
            self.A = self.__snapShots__()
            pd.DataFrame(self.A).to_csv(name, index=False, header=False)

    @staticmethod
    def initialCondition(x):
        return 6*np.sin(np.pi*x)
    
    def __snapShots__(self):
        U = np.zeros((self.nt, self.nx))
        U[0,:] = self.initialCondition(np.linspace(0, 1, self.nx))
        U[0,0], U[0,-1] = 0, 0
        dt = self.T/self.nt
        dx = 1/self.nx

        for t in range(self.nt-1):
            for x in range(1, self.nx-1):
                U[t+1,x] = U[t, x] + self.k*dt/(dx**2)*(U[t,x+1] - 2*U[t,x] + U[t,x-1])

        return U

    def solution(self,x,t):
        return 6*np.sin(np.pi*x)*np.exp(-self.k*np.pi**2*t)

class POD:
    """
    POD class

    This class is used to compute the POD method.

    Attributes
    ----------
    EDP : class
        class of the partial differential equation
    
    solution : numpy array
        solution of the partial differential equation if known

    """
    def __init__(self, EDP, solution=None):
        self.A = EDP.A
        self.solution = solution if solution(0,0) is not None else None
        self.x, self.t = np.linspace(0, 1, EDP.nx), np.linspace(0, EDP.T, EDP.nt)
        self.X, self.T = np.meshgrid(self.x, self.t)
        self.__compute_svd__()
    
    def __compute_svd__(self):
        self.U, self.S, self.VT = np.linalg.svd(self.A)

    def aproximate(self, rank):
        A_ = self.U[:,:rank] @ np.diag(self.S[:rank]) @ self.VT[:rank, :]
        return A_
    
    def __plot__(self, fig, title, n, i, A):
        ax = fig.add_subplot(1, n, i, projection='3d')
        ax.plot_surface(self.T, self.X, A, cmap='viridis')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("t")
        ax.set_ylabel("x")

    def plot_aproximations(self, rank=3):
        # Calculate the aproximations
        A_ = self.aproximate(rank)

        n = 2 if self.solution is None else 3

        # Create subplots
        fig = plt.figure(figsize=(4*n,8), constrained_layout=True)
        fig.set_constrained_layout_pads(h_pad=0.05, w_pad=0.05, hspace=0.2, wspace=0.2)

        if self.solution: self.__plot__(fig, "Solution", n, 1, self.solution(self.X, self.T))

        self.__plot__(fig, "Original matrix", n, n-1, self.A)
        self.__plot__(fig, f"Rank {rank} approximation", n, n, A_)

        fig.suptitle(f"POD Approximations (Rank = {rank})", fontsize=16, fontweight="bold")
        plt.show()

    def plot_singular_values(self):
        plt.plot(self.S, "o-", color="#1f77b4")
        plt.yscale("log")
        plt.xlabel("Rank")
        plt.ylabel("$\sigma_{i}$")
        plt.title("Singular values", fontsize=16, fontweight="bold")
        plt.show()

    def plot_modal_contributions(self, rank=3):
        # C  =  np.diag(self.S[:rank]) @ self.U.T[:rank, :] # is another way of cumputing C due to orthonormality
        C  = self.VT[:rank, :] @ self.A.T
        
        plt.figure(figsize=(8, 4))
        for i, c_i in enumerate(C): plt.plot(self.t, c_i, "-", label=f"Mode {i+1}")
        plt.xlabel('t')
        plt.ylabel('modal coordinates')
        plt.title('The modal contributions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_errors(self, rank=3):
        plt.figure(figsize=(6, 4))
        errors = []
        normZ = np.linalg.norm(self.A)

        if self.solution:
            SOL = self.solution(self.X, self.T)
            errors_sols = []
            normZ = np.linalg.norm(SOL)

        for r in range(rank):
            C  = self.VT[:r+1, :] @ self.A.T
            A_reconstructed = C.T @ self.VT[:r+1, :] # A linear combination of the coefficients and the modes (C(t)*V(x))
            error_i = np.linalg.norm(self.A - A_reconstructed)/normZ
            errors.append(error_i)

            if self.solution: errors_sols.append(np.linalg.norm(SOL - A_reconstructed)/normZ)
        
        plt.plot(np.arange(1, rank+1), errors, "o-", color="#209c49", label="Matrix aproximation error")
        if self.solution: plt.plot(np.arange(1, rank+1), errors_sols, "o-", color="#ff7f0e", label="Function aproximation error")
        plt.xlabel("Rank")
        plt.ylabel("Error")
        plt.title("Relative error of the aproximation", fontsize=16, fontweight="bold")
        plt.legend()
        plt.show()

def main():
    burger = BURGER(0.025, 1, 1000, 40)
    pod = POD(burger, burger.solution)

    pod.plot_aproximations(10)
    pod.plot_singular_values()
    pod.plot_modal_contributions(10)
    pod.plot_errors(10)

if __name__ == "__main__":
    main()