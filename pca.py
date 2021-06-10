from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    #Load the dataset from a provided .npy file, re-center it around the origin and return it as a NumPy array of floats
    x = np.load(filename)
    m = np.mean(x, axis = 0)
    retval = x-m
    return retval
    
def get_covariance(dataset):
    #calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
    sigma = np.dot(np.transpose(dataset), dataset)
    return sigma/(dataset.shape[0]-1)
    
def get_eig(S, m):
    #perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array) with the largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the corresponding eigenvectors as columns
    #upper = S.shape[0]
    upper = len(S)
    w, U_t = eigh(S, eigvals = (upper-m,upper-1))
    w_t = np.diag(w[::-1])
    #Lambda = w_t.transpose()[::-1].transpose()[::-1]
    U = U_t.transpose()[::-1].transpose()
    return w_t, U
    
def get_eig_perc(S, perc):
    #similar to get_eig, but instead of returning the first m, return all eigenvalues and corresponding eigenvectors in similar format as get_eig that explain more than perc % of variance
    w, u = eigh(S)
    w_perc = w[(w/sum(w))>perc]
    m = len(w_perc)
    if m>0:
        return get_eig(S, m)
    else:
        return np.empty(0), np.empty(0)
    
def project_image(img, U):
    #project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
    alphas = np.dot(img, U)
    return np.dot(alphas, U.T)
    
def display_image(orig, proj):
    #use matplotlib to display a visual representation of the original image and the projected image side-by-side
    o_rs = orig.reshape(32,32).T
    p_rs = proj.reshape(32,32).T
    fig, (ax1,ax2) = plt.subplots(1,2)
    a = ax1.imshow(o_rs, aspect = "equal")
    b = ax2.imshow(p_rs, aspect = "equal")
    fig.colorbar(a,ax=ax1)
    fig.colorbar(b,ax=ax2)
    ax1.set_title("Original")
    ax2.set_title("Projection")
    plt.show()