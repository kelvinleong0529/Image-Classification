import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class TSNEVisual:
    # doc: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    def __init__(self, tsne_obj: TSNE = None):
        """
        :param tsne_obj:        <Optional>
                                    if given, t-SNE will be operated using the given transformer,
                                    otherwise, using the default one:
                                        n_components=2,init="pca",random_state=0
        """
        if tsne_obj:
            self.tsne_obj = tsne_obj
        else:
            self.tsne_obj = TSNE(n_components=2, init='pca', random_state=0)

    def _transform_2_embed(self, in_features: np.ndarray) -> np.ndarray:
        """
        Fit the given feature into an embedded space
        :param in_features:      given feature
        :return:                transformed output
        """
        print("[TSNE] Start Transformation of Feature Shape", in_features.shape)
        _embed = self.tsne_obj.fit_transform(in_features)
        print("[TSNE] \tTransformation Completed")
        return _embed

    def get_transform_embed(self, in_features: np.ndarray) -> np.ndarray:
        """
        API: Fit the given features into an embedded space and return that transformed output
        :param in_features:     given features
        :return:                transformed output
        """
        return self._transform_2_embed(in_features=in_features)

    @staticmethod
    def _plot_embed_2d(in_embeds: np.ndarray, in_labels: np.ndarray,
                       norm: bool = True) -> plt:
        """
        Plot (2d) the given embeddings based on the given labels
            and return the plt object
        :param in_embeds:       given embeddings of all samples
        :param in_labels:       given labels of all samples
        :param norm:            whether to normalize the given embeddings:
                                    (embed - embed_min) / (embed_max - embed_min)
        :return:                <matplotlib.pyplot> object
        """
        x_min, x_max = np.min(in_embeds, 0), np.max(in_embeds, 0)
        if norm:
            in_embeds = (in_embeds - x_min) / (x_max - x_min)
        max_label = np.max(in_labels)
        plt.scatter(in_embeds[:, 0], in_embeds[:, 1], c=in_labels,
                    s=1, cmap=plt.cm.get_cmap("jet", max_label))
        plt.colorbar(ticks=range(max_label))
        # plt.clim(-0.5, 9.5)
        # plt.xticks([])
        # plt.yticks([])
        # plt.title(title)
        return plt

    def plot_embed_2d(self, in_embeds: np.ndarray, in_labels: np.ndarray,
                      norm: bool = True) -> plt:
        """
        API: Plot (2d) the given embedding based on the given labels
            and return the plt object
        :param in_embeds:       given embedding
        :param in_labels:       given labels of all samples
        :param norm:            whether to normalize the given embeddings:
                                    (embed - embed_min) / (embed_max - embed_min)
        :return:                <matplotlib.pyplot> object
        """
        return self._plot_embed_2d(in_embeds=in_embeds, in_labels=in_labels)

    def plot_feature_2d(self, in_features: np.ndarray, in_labels: np.ndarray,
                        norm: bool = True) -> plt:
        """
        API: Fit the given feature into an embedded space, visualize (2d) based on labels
            and return the plt object
        :param in_features:     given feature
        :param in_labels:       given labels of all samples
        :param norm:            whether to normalize the given embeddings:
                                    (embed - embed_min) / (embed_max - embed_min)
        :return:                <matplotlib.pyplot> object
        """
        _embed = self._transform_2_embed(in_features=in_features)
        return self._plot_embed_2d(in_embeds=_embed, in_labels=in_labels)


if "__main__" == __name__:
    _test_num_sample = 1000
    _test_feature = np.random.normal(size=(_test_num_sample, 10))  # shape (sample, dim)=(num_sample, 10)
    _test_tsne_visual = TSNEVisual(tsne_obj=None)
    _test_embed = _test_tsne_visual.get_transform_embed(in_features=_test_feature)
    print(_test_embed.shape)  # shape (sample, dim)=(num_sample, 2)
    _test_embed_plt = _test_tsne_visual.plot_embed_2d(
        in_embeds=_test_embed,
        in_labels=np.random.randint(low=0, high=10, size=_test_num_sample), norm=False)
    _test_embed_plt.show()
