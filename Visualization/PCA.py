from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class myPCA:
    def __init__(self, data, out_dim = 2):
        # data shape: [samples, dimension], numpy matrix
        self.n_sample = data.shape[0]
        self.in_dim = data.shape[1]
        self.out_dim = out_dim
        mean = np.array([np.mean(X[:, i]) for i in range(self.in_dim)])
        self.data = data - mean

    def dim_reduction(self):
        """
        Reduce the dimension from input_dim to out_dim
        [samples, input_dim] ---> [samples, output_dim]
        :return: reduced matrix
        """
        xTx = np.dot(self.data.T, self.data) #[in_dim, in_dim]

        eig_value, eig_vector = np.linalg.eig(xTx)
        self.eig_value = eig_value

        # sort the eigen value from high to low
        sorted_eigvalue_idx = sorted(range(len(eig_value)), key=lambda k: eig_value[k],reverse=True)

        # pick the top k columns in eigenvector corresponding to sorted eigenvalue
        feature_vector = np.array([eig_vector[:,i] for i in sorted_eigvalue_idx[:self.out_dim]])
        # [out_dim, n_sample]

        reduced_data = np.dot(self.data, feature_vector.T) #[n_sample, out_dim]
        self.reduced_data = reduced_data

        return reduced_data

    def draw_eigenvalue(self,filename):
        """
        Draw the curve of sorted eigenvalue
        :param filename: the filename for saving the picture
        """
        sorted_eigvalue = sorted(self.eig_value, reverse=True)
        x = [i for i in range(len(sorted_eigvalue))]
        plt.plot(x, sorted_eigvalue)
        plt.title('Eigenvalue Curve')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.legend()
        plt.savefig(filename)
        plt.show()

    def draw_dataset(self,filename,dataset):
        """
        将数据集中的图片画在二维平面上。前提是需要将 PCA 的输出维度设为 2，把每张图转化为二维坐标
        :param filename: the filename for saving the picture
        :param dataset: image matrix with size [n_samples, High, Width, Channel]
        """
        assert dataset.shape[0] == self.n_sample
        assert self.out_dim == 2
        assert len(dataset.shape) == 4

        x = []
        y = []
        for coordinate in self.reduced_data:
            x.append(coordinate[0])
            y.append(coordinate[1])

        fig, ax = plt.subplots()
        ax.scatter(x, y)

        for x0, y0, data in zip(x, y, dataset):
            ab = AnnotationBbox(OffsetImage(data), (x0, y0), frameon=False)
            ax.add_artist(ab)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(filename)
        plt.show()

if __name__ == "__main__":
    #这里用 project 1 的 SVHN 数据集作为例子
    m = loadmat("./dataset/test_32x32.mat")
    n_sample = 50
    images_matrix = m["X"][:, :, :,0:n_sample]
    X = images_matrix.reshape((32*32*3,n_sample))
    X = X.T
    dim = 2

    input_img = np.array([images_matrix[:, :, :,i] for i in range(n_sample)])

    res = myPCA(X,dim)
    res.dim_reduction()
    res.draw_eigenvalue("eigenvalue.png")
    res.draw_dataset("PCA_dataset.png",input_img)
