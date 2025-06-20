"""
POD numerical example

This script is a numerical example of the POD method.
The POD method is used to reduce the dimensionality of a dataset.

@Author: Miquel Baztán
@Date: 17/06/2025-18/06/2025
"""

import numpy as np
import matplotlib.pyplot as plt

def z1(x, t):
    return np.exp(-np.abs((x - 0.5)*(t - 1))) + np.sin(x*t)

EPS = 0.01

def z(x, t):
    return np.sin(1/(x+t+EPS))

class POD:
    """
    POD class

    This class is used to compute the POD method.

    Attributes
    ----------
    f : function
        function to approximate
    x_bounds : list
        bounds of the x axis (x_bounds[0] <= x <= x_bounds[1])
    t_bounds : list
        bounds of the t axis (t_bounds[0] <= t <= t_bounds[1])
    nx_snapshots : int
        number of snapshots in the x axis
    nt_snapshots : int
        number of snapshots in the t axis

    """
    def __init__(self, f, x_bounds, t_bounds, nx_snapshots, nt_snapshots):
        self.f = f
        self.x_bounds = x_bounds
        self.t_bounds = t_bounds

        self.__compute_snapshots__(nx_snapshots, nt_snapshots)
        self.__compute_svd__()
    
    def __compute_snapshots__(self, nx_snapshots, nt_snapshots):
        self.x = np.linspace(self.x_bounds[0], self.x_bounds[1], nx_snapshots)
        self.t = np.linspace(self.t_bounds[0], self.t_bounds[1], nt_snapshots)

        [self.X, self.T] = np.meshgrid(self.x, self.t)
        
        self.Z = self.f(self.X, self.T)
    
    def __compute_svd__(self):
        self.U, self.S, self.VT = np.linalg.svd(self.Z)

    def aproximate(self, rank):
        zz = self.U[:,:rank] @ np.diag(self.S[:rank]) @ self.VT[:rank, :]
        return zz
    
    def plot_aproximations(self, rank=3):
        # Calculate the aproximations
        ZZ = [self.aproximate(r+1) for r in range(rank)]
        ZZ.insert(0, self.Z)

        # Create subplots
        fil = int(np.floor(np.sqrt(rank+1)))
        col = int(np.ceil((rank+1)/fil))

        fig = plt.figure(figsize=(4*max(fil,col), 4*max(fil,col)), constrained_layout=True)
        fig.set_constrained_layout_pads(h_pad=0.05, w_pad=0.05, hspace=0.2, wspace=0.2)

        for i in range(fil):
            for j in range(col):
                idx = i * col + j
                if idx >= len(ZZ): break
                ax = fig.add_subplot(fil, col, idx+1, projection='3d')
                ax.plot_surface(self.X, self.T, ZZ[idx], cmap='viridis')

                title = "Original" if idx == 0 else f"Rank {idx} approximation"
                ax.set_title(title, fontsize=12)

            else: continue
            break
        
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
        C  = self.VT[:rank, :] @ self.Z.T
        
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
        normZ = np.linalg.norm(self.Z)

        for r in range(rank):
            C  = self.VT[:r+1, :] @ self.Z.T
            Z_reconstructed = C.T @ self.VT[:r+1, :] # A linear combination of the coefficients and the modes (C(t)*V(x))
            error_i = np.linalg.norm(self.Z - Z_reconstructed)/normZ
            errors.append(error_i)
        
        plt.plot(np.arange(1, rank+1), errors, "o-", color="#209c49")
        plt.xlabel("Rank")
        plt.ylabel("Error")
        plt.title("Relative error of the aproximation", fontsize=16, fontweight="bold")
        plt.show()

def main():
    p = POD(z, [0, 1], [0,2], 25, 50)

    p.plot_aproximations(5)
    p.plot_singular_values()
    p.plot_modal_contributions(5)
    p.plot_errors(5)

if __name__ == "__main__":
    main()