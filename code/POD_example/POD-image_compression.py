"""
POD image compression

This script is an example of the POD method applied to image compression.

@Author: Miquel Baztán
@Date: 19/06/2025
"""

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

PATH = "POD_example/data"

class POD:
    """
    POD class

    This class is used to compute the POD method.

    Attributes
    ----------
    img_name : str
        name of the image

    mode : str
        mode of the image ("RGB" or "L")
    """
    def __init__(self, img_name, mode = "RGB"):
        self.mode = mode
        self.img_name = img_name

        img = Image.open(f"{PATH}/{img_name}.jpg")
        format_img = img.convert(mode)
        self.A = np.array(format_img)
        
        if mode == "RGB":
            self.A = self.A/255

            self.R, self.G, self.B = self.A[:,:,0], self.A[:,:,1], self.A[:,:,2]
        else:
            self.A = self.A/np.max(self.A)

        self.__compute_svd__()
    
    def __compute_svd__(self):
        if self.mode == "RGB":
            self.UR, self.SR, self.VTR = np.linalg.svd(self.R)
            self.UG, self.SG, self.VTG = np.linalg.svd(self.G)
            self.UB, self.SB, self.VTB = np.linalg.svd(self.B)
        else:
            self.U, self.S, self.VT = np.linalg.svd(self.A)

    def aproximate(self, rank):
        if self.mode == "RGB":
            zz = np.zeros((self.A.shape[0], self.A.shape[1], 3))
            zz[:,:,0] = self.UR[:,:rank] @ np.diag(self.SR[:rank]) @ self.VTR[:rank, :]
            zz[:,:,1] = self.UG[:,:rank] @ np.diag(self.SG[:rank]) @ self.VTG[:rank, :]
            zz[:,:,2] = self.UB[:,:rank] @ np.diag(self.SB[:rank]) @ self.VTB[:rank, :]
            return zz

        zz = self.U[:,:rank] @ np.diag(self.S[:rank]) @ self.VT[:rank, :]
        return zz

    def plot_aproximations(self, rank=3):
        # Calculate the aproximation
        A_ = self.aproximate(rank)

        # Plot the images [real, aproximation]
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].imshow(self.A, cmap="gray" if self.mode == "L" else None)
        axs[0].set_title("Original")

        axs[1].imshow(A_, cmap="gray" if self.mode == "L" else None)
        axs[1].set_title(f"Rank {rank} approximation")

        fig.suptitle(f"POD Approximation (Rank = {rank})", fontsize=16, fontweight="bold")
        plt.show()

    def plot_singular_values(self):
        if (self.mode == "RGB"):
            plt.plot(self.SR, "o-", color="#ae2727")
            plt.plot(self.SG, "o-", color="#41b81d")
            plt.plot(self.SB, "o-", color="#2c3ba0")
        else:
            plt.plot(self.S, "o-", color="#606060ff")
            
        plt.yscale("log")
        plt.xlabel("Rank")
        plt.ylabel("$\sigma_{i}$")
        plt.title("Singular values", fontsize=16, fontweight="bold")
        plt.show()

    def __plot_modal_contributions__(self, rank, VT, A):
        # C  =  np.diag(self.S[:rank]) @ self.U.T[:rank, :] # is another way of cumputing C due to orthonormality
        C  = VT[:rank, :] @ A.T
        
        plt.figure(figsize=(8, 4))
        for i, c_i in enumerate(C): plt.plot(np.arange(self.A.shape[0]), c_i, "-", label=f"Mode {i+1}")
        plt.xlabel('t')
        plt.ylabel('modal coordinates')
        plt.title('The modal contributions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_modal_contributions(self, rank=3):
        if (self.mode == "RGB"):
            self.__plot_modal_contributions__(rank, self.VTR, self.R)
            self.__plot_modal_contributions__(rank, self.VTG, self.G)
            self.__plot_modal_contributions__(rank, self.VTB, self.B)
        else:
            self.__plot_modal_contributions__(rank, self.VT, self.A)

    def plot_errors(self, rank=3):
        plt.figure(figsize=(6, 4))
        errors = []
        normZ = np.linalg.norm(self.A)

        for r in range(rank):
            if (self.mode == "RGB"):
                CR = self.VTR[:r+1, :] @ self.R.T
                ZR_reconstructed = CR.T @ self.VTR[:r+1, :]

                CG = self.VTG[:r+1, :] @ self.G.T
                ZG_reconstructed = CG.T @ self.VTG[:r+1, :]

                CB = self.VTB[:r+1, :] @ self.B.T
                ZB_reconstructed = CB.T @ self.VTB[:r+1, :]

                Z_reconstructed = np.zeros((self.A.shape[0], self.A.shape[1], 3))
                Z_reconstructed[:,:,0] = ZR_reconstructed
                Z_reconstructed[:,:,1] = ZG_reconstructed
                Z_reconstructed[:,:,2] = ZB_reconstructed
            else:
                C  = self.VT[:r+1, :] @ self.A.T
                Z_reconstructed = C.T @ self.VT[:r+1, :] # A linear combination of the coefficients and the modes (C(t)*V(x))

            error_i = np.linalg.norm(self.A - Z_reconstructed)/normZ
            errors.append(error_i)
        
        plt.plot(np.arange(1, rank+1), errors, "-", color="#209c49")
        plt.xlabel("Rank")
        plt.ylabel("Error")
        plt.title("Relative error of the aproximation", fontsize=16, fontweight="bold")
        plt.show()

def main():
    p = POD("rgb-sky", "RGB")
    p.plot_aproximations(100)
    p.plot_singular_values()
    p.plot_modal_contributions(100)
    p.plot_errors(100)

if __name__ == "__main__":
    main()