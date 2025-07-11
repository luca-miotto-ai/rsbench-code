"""
Collection of post-hoc concept explainer methods

Adapted from Jonathan Crabbe's library for 
Concept Activation Regions (CARs): 
https://github.com/JonathanCrabbe/CARs
"""

import torch
import optuna
import logging
import numpy as np
import torch.nn.functional as F

from typing import Callable
from abc import ABC, abstractmethod

from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm._base import BaseSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import permutation_test_score

class PostHocConceptExplainer(ABC):
    """
    An abstract class that contains the interface 
    for a generic post-hoc concept explainer
    """

    def __init__(self, classifier: ClassifierMixin, device: torch.device, batch_size: int = 50):
        """
        Parameters
        ----------
        classifier: ClassifierMixin
            Concept classifier (e.g. SGDClassifier, SVC, etc.)
        device: torch.device
            Device to run the model on (e.g. 'cuda' or 'cpu')
        batch_size: int, optional
            Batch size for processing the data (default is 50)
        """
        if not isinstance(device, torch.device):
            raise TypeError("Device must be a torch device")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        if not isinstance(classifier, ClassifierMixin):
            raise TypeError("Classifier must be a subclass of ClassifierMixin")
        if not PostHocConceptExplainer._hasmethod(classifier, "fit"):
            raise ValueError("Classifier must implement the fit method")
        if not PostHocConceptExplainer._hasmethod(classifier, "predict"):
            raise ValueError("Classifier must implement the predict method")
        self.representations = None
        self.presence = None
        self.significance = {}
        self.device = device
        self.batch_size = batch_size
        self.classifier = classifier

    @staticmethod
    def _hasmethod(obj: object, attr: str) -> bool:
        """
        Check if the object has the given method

        Parameters
        ----------
        obj: object
            Object to check
        attr: str
            Name of the method to check

        Returns
        -------
            If the object has the method
        """
        return hasattr(obj, attr) and callable(getattr(obj, attr))

    @staticmethod
    def _reshape_2d(representations: np.ndarray) -> np.ndarray:
        """Reshape N-dimensional representations into 2D matrix"""
        return representations.reshape(
            representations.shape[0], 
            np.prod(representations.shape[1:])
        )

    @staticmethod
    def _reshuffle(representations: np.ndarray, presence: np.ndarray) -> tuple:
        """Randomly reshuffle positive/negative examples"""
        reindexing = np.random.permutation(len(representations))
        representations_ = representations[reindexing]
        presence_ = presence[reindexing]
        return representations_, presence_

    def fit(self, representations: np.ndarray, presence: np.ndarray):
        """
        Fit the concept classifier to the dataset (concept representations and presence)

        Parameters
        ----------
        representations: np.ndarray
            Latent representations of the training examples illustrating the concept
        presence: np.ndarray
            Boolean array indicating presence or absence of the concept in each example
        """
        if representations.shape[0] != presence.shape[0]:
            raise ValueError("Representations length does not match presence")
        self.representations = representations
        self.presence = presence
        repr_, pres_ = PostHocConceptExplainer._reshuffle(representations, presence)
        repr_ = PostHocConceptExplainer._reshape_2d(representations)
        self.classifier.fit(repr_, pres_)

    def predict(self, representations: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept given the latent representations
        
        Parameters
        ----------
        representations: np.ndarray
            Latent representations of the test examples illustrating the concept
        
        Returns
        -------
        Boolean array indicating presence or absence of the concept
        """
        repr_ = PostHocConceptExplainer._reshape_2d(representations)
        return self.classifier.predict(repr_)

    def get_representations(self, positive: bool = True) -> np.ndarray:
        """
        Get the latent representations of the concept

        Parameters
        ----------
        positive: bool, optional
            Get the concept representations of the positive/negative set

        Returns
        -------
        Either positive or negative representations of the concept
        """
        if self.representations is None:
            raise ValueError("No latent representations available yet!")
        return self.representations[self.presence == int(positive)]

    @abstractmethod
    def concept_importance(self, representations: np.ndarray) -> np.ndarray:
        """
        Predicts the relevance of a concept given the latent representations
        
        Parameters
        ----------
        representations: np.ndarray
            Latent representations of the test examples
        
        Returns
        -------
        Array of concept importance scores for each example
        """

    def significant(self, test='permutation', alpha=0.05,  n_jobs=1, **kwargs) -> bool:
        """
        Computes the p-value of the given significance test applied on the previously calculated \\
        concept presence and decides if significant according to the given significance level `alpha`

        Parameters
        ----------
        test: str, optional
            Type of significance test to perform
        alpha: float, optional
            Significance level of the test
        n_jobs: int, optional
            Number of jobs to run in parallel

        Returns
        -------
        If the concept is statistically significant, up to significance level

        Remarks
        -------
        - Tests currently implemented: `permutation`
        - Significace tests are performed on the concept presence wrt stored representations
        - The same classifier is used for significance test as the one used to classify concepts
        - Optional arguments for each implemented test:
            - `permutation` allows for optional argument `n_perm` to specify number of permutations
        """
        if test == 'permutation':
            n_perm = kwargs.get('n_perm', 100)
            repr_ = PostHocConceptExplainer._reshape_2d(self.representations)
            _, _, p_value = permutation_test_score(
                self.classifier,
                repr_,
                self.presence,
                n_permutations=n_perm,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Invalid significance test \"{test}\"")
        return bool(p_value < alpha)

class CAV(PostHocConceptExplainer):
    """
    Concept Activation Vectors (CAV) post-hoc concept explainer

    References:
        Kim, Been et al. "Interpretability Beyond Feature Attribution: 
        Quantitative Testing with Concept Activation Vectors (TCAV)." 
        International Conference on Machine Learning (2017).
    """
    
    def __init__(self, classifier: LinearClassifierMixin, device: torch.device, batch_size: int = 50):
        super().__init__(classifier, device, batch_size)
        if not isinstance(classifier, LinearClassifierMixin):
            raise TypeError("Classifier must be a subclass of LinearClassifierMixin")

    def concept_importance(self,
        representations: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        repr_to_output: Callable
    ) -> np.ndarray:
        """
        Predicts the relevance of a concept given the latent representations.\\
        Note that, concept importance for CAV is akin to concept sensitivity.
        
        Parameters
        ----------
        representations: np.ndarray
            Latent representations of the test examples
        labels: np.ndarray
            Labels associated to the representations
        num_classes: int
            Total number of classes for one-hot encoding
        repr_to_output: Callable 
            Black-box mapping the representation space into the output space

        Returns
        -------
        Concept scores for each example (i.e. for each representation)
        """
        labels_ = torch.from_numpy(labels).to(self.device)
        one_hot_labels = F.one_hot(labels_, num_classes).to(self.device)
        latent_repr = torch.from_numpy(representations).to(self.device).requires_grad_()
        outputs = repr_to_output(latent_repr)
        grads = torch.autograd.grad(outputs, latent_repr, grad_outputs=one_hot_labels)[0]
        cav = torch.tensor(self.classifier.coef_).to(self.device).float()
        if len(grads.shape) > 2:
            grads = grads.flatten(start_dim=1)
        if len(cav.shape) > 2:
            cav = cav.flatten(start_dim=1)
        return torch.einsum("ij,ij->i", cav, grads).detach().cpu().numpy()

class CAR(PostHocConceptExplainer):
    """
    Concept Activation Regions (CAR) post-hoc concept explainer

    References
    ----------
        Crabbé, Jonathan, and Mihaela van der Schaar. \\
        "Concept Activation Regions: A Generalized Framework \\
        for Concept-Based Explanations." Advances in Neural \\
        Information Processing Systems (2022).
    """
    
    def __init__(self, 
            classifier: BaseSVC, 
            device: torch.device, 
            batch_size: int = 100, 
            kernel_width: float = 1.0
        ):
        super().__init__(classifier, device, batch_size)
        if not isinstance(classifier, BaseSVC):
            raise TypeError("Classifier must be a subclass of BaseSVC")
        self.kernel_width = kernel_width

    def concept_importance(self, representations: np.ndarray, to_array: bool = True) -> np.ndarray|torch.Tensor:
        pos_density = self.concept_density(representations, True)
        neg_density = self.concept_density(representations, False)
        if to_array:
            return (pos_density - neg_density).detach().cpu().numpy()
        else:
            return pos_density - neg_density

    def concept_density(self, representations: np.ndarray, positive_set: bool) -> torch.Tensor:
        """
        Computes the concept density for the given latent representations

        Parameters
        ----------
        representations: np.ndarray
            Latent representations for which the concept density should be evaluated
        positive_set: bool
            If True, only compute for the positive set. Otherwise, only for the negative.

        Returns
        -------
        Density of the latent representations under the relevant concept set
        """
        kernel = self._kernel_function()
        latent_reps = torch.from_numpy(representations).to(self.device)
        concept_reps = torch.from_numpy(self.get_representations(positive_set)).to(self.device)
        density = kernel(concept_reps, latent_reps).mean(dim=0)
        return density

    def _kernel_function(self) -> Callable:
        """
        Get the kernel function underlying the CAR

        Returns
        -------
        Kernel function as a callable with arguments (h1, h2)
        """
        kernel_type = self.classifier.kernel
        if kernel_type == "rbf":
            latent_dim = self.representations.shape[-1]
            epsilon = 1 / (latent_dim * self.kernel_width)
            return CAR._gaussian_rbf(epsilon)
        elif kernel_type == "linear":
            return lambda h1, h2: torch.einsum("abi, abi -> ab", h1.unsqueeze(1), h2.unsqueeze(0))
        else:
            raise ValueError(f"Unknown kernel type {kernel_type}. " + 
                    "Currently supported types are 'rbf' and 'linear'.")

    @staticmethod
    def _gaussian_rbf(epsilon: float = 1.0) -> Callable:
        """
        Get the Gaussian RBF kernel function

        Parameters
        ----------
        epsilon: float, optional
            Scale parameter for reshaping (default is 1.0)

        Returns
        -------
        Gaussian RBF kernel function as a callable with arguments (h1, h2)
        """
        return lambda h1, h2: torch.exp(
            -torch.sum((epsilon * (h1.unsqueeze(1) - h2.unsqueeze(0))) ** 2, dim=-1)
        )

    def tune_kernel_width(self, representations: np.ndarray, presence: np.ndarray):
        """
        Tune the kernel width to achieve good training 
        classification accuracy with a Parzen classifier

        Parameters
        ----------
        representations: np.ndarray
            Latent representations of the training examples illustrating the concept
        presence: np.ndarray
            Boolean array indicating presence or absence of the concept in each example
        
        """
        super().fit(representations, presence)

        def train_acc(trial):
            kernel_width = trial.suggest_float("kernel_width", 0.1, 50)
            self.kernel_width = kernel_width
            density = []
            for reps_batch in np.split(representations, self.batch_size):
                density.append(self.concept_importance(reps_batch))
            density = np.concatenate(density)
            return accuracy_score((density > 0).astype(int), presence)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(train_acc, n_trials=1000)
        self.kernel_width = study.best_params["kernel_width"]
        logging.info(
            f"Optimal kernel width {self.kernel_width:.3g} " + 
            f"with training accuracy {study.best_value:.2g}"
        )

    # def fit_cv(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
    #     """
    #     Fit the concept classifier to the dataset (latent_reps, concept_labels) by tuning the kernel width
    #     Args:
    #         concept_reps: latent representations of the examples illustrating the concept
    #         concept_labels: labels indicating the presence (1) or absence (0) of the concept
    #     """
    #     super(CAR, self).fit(concept_reps, concept_labels)

    #     X_train, X_val, y_train, y_val = train_test_split(
    #         concept_reps,
    #         concept_labels,
    #         test_size=int(0.3 * len(concept_reps)),
    #         stratify=concept_labels,
    #     )

    #     def objective(trial: optuna.Trial) -> float:
    #         kernel = trial.suggest_categorical(
    #             "kernel", ["linear", "poly", "rbf", "sigmoid"]
    #         )
    #         gamma = trial.suggest_loguniform("gamma", 1e-3, 1e3)
    #         C = trial.suggest_loguniform("C", 1e-3, 1e3)
    #         classifier = SVC(kernel=kernel, gamma=gamma, C=C)
    #         classifier.fit(X_train, y_train)
    #         return accuracy_score(y_val, classifier.predict(X_val))

    #     optuna.logging.set_verbosity(optuna.logging.WARNING)
    #     study = optuna.create_study(direction="maximize")
    #     study.optimize(objective, n_trials=200, show_progress_bar=True)
    #     best_params = study.best_params
    #     self.classifier = SVC(**best_params)
    #     self.classifier.fit(concept_reps, concept_labels)
    #     self.kernel_width = best_params["gamma"]
    #     logging.info(
    #         f"Optimal hyperparameters {best_params} with validation accuracy {study.best_value:.2g}"
    #     )

    def concept_sensitivity(self,
        representations: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        repr_to_output: Callable
    ) -> np.ndarray:
        """
        Predicts the relevance of a concept given the latent representations.\\
        Note that, concept importance for CAV is akin to concept sensitivity.
        
        Parameters
        ----------
        representations: np.ndarray
            Latent representations of the test examples
        labels: np.ndarray
            Labels associated to the representations
        num_classes: int
            Total number of classes for one-hot encoding
        repr_to_output: Callable 
            Black-box mapping the representation space into the output space

        Returns
        -------
        Concept scores for each example (i.e. for each representation)
        """
        labels_ = torch.from_numpy(labels).to(self.device)
        one_hot_labels = F.one_hot(labels_, num_classes).to(self.device)
        latent_repr = torch.from_numpy(representations).to(self.device).requires_grad_()
        outputs = repr_to_output(latent_repr)
        grads = torch.autograd.grad(outputs, latent_repr, grad_outputs=one_hot_labels)[0]
        densities = self.concept_importance(latent_repr, to_array=False).view((-1, 1)) # FIXME: integrate np and torch flows!
        cavs = torch.autograd.grad(
            densities,
            latent_repr,
            grad_outputs=torch.ones((len(densities), 1)).to(self.device),
        )[0]

        if len(grads.shape) > 2:
            grads = grads.flatten(start_dim=1)
        if len(cavs.shape) > 2:
            cavs = cavs.flatten(start_dim=1)
        return torch.einsum("bi,bi->b", cavs, grads).detach().cpu().numpy()
