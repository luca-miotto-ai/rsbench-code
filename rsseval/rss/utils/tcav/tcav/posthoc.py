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
        if representations.shape[0] == presence.shape[0]:
            raise ValueError("Representations length does not match presence")
        self.representations = representations
        self.presence = presence
        self.classifier.fit(representations, presence)

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
        return self.classifier.predict(representations)

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

    @abstractmethod
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
            _, _, p_value = permutation_test_score(
                self.classifier,
                self.representations,
                self.presence,
                n_permutations=n_perm,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Invalid significance test \"{test}\"")
        return p_value < alpha

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

    def concept_importance(
        self,
        representations: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        repr_to_output: Callable
    ) -> np.ndarray:
        """
        Predicts the relevance of a concept given the latent representations
        
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
        one_hot_labels = F.one_hot(labels, num_classes).to(self.device)
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

    References:
        Crabbé, Jonathan, and Mihaela van der Schaar. 
        "Concept Activation Regions: A Generalized Framework 
        for Concept-Based Explanations." Advances in Neural 
        Information Processing Systems (2022).
    """
    
    def __init__(
        self,
        classifier: BaseSVC,
        device: torch.device,
        batch_size: int = 100,
        kernel: str = "rbf",
        kernel_width: float = None,
    ):
        super().__init__(classifier, device, batch_size)
        if not isinstance(classifier, BaseSVC):
            raise TypeError("Classifier must be a subclass of BaseSVC")
        self.kernel = kernel
        self.kernel_width = kernel_width

    def fit(self, classifier: ClassifierMixin, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            concept_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(CAR, self).fit(concept_reps, concept_labels)
        classifier = SVC(kernel=self.kernel)
        classifier.fit(concept_reps, concept_labels)
        self.classifier = classifier

    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        return self.classifier.predict(latent_reps)

    def concept_importance(self, latent_reps: torch.Tensor) -> torch.Tensor:
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts scores for each example
        """
        pos_density = self.concept_density(latent_reps, True)
        neg_density = self.concept_density(latent_reps, False)
        return pos_density - neg_density

    def permutation_test(
        self,
        concept_reps: np.ndarray,
        concept_labels: np.ndarray,
        n_perm: int = 100,
        n_jobs: int = -1,
    ) -> float:
        """
        Computes the p-value of the concept-label permutation test
        Args:
            concept_labels: concept labels indicating the presence (1) or absence (0) of the concept
            concept_reps: representation of the examples
            n_perm: number of permutations
            n_jobs: number of jobs running in parallel

        Returns:
            p-value of the statistical significance test
        """
        classifier = SVC(kernel=self.kernel)
        score, permutation_scores, p_value = permutation_test_score(
            classifier,
            concept_reps,
            concept_labels,
            n_permutations=n_perm,
            n_jobs=n_jobs,
        )
        return p_value

    def get_kernel_function(self) -> Callable:
        """
        Get the kernel funtion underlying the CAR
        Returns: kernel function as a callable with arguments (h1, h2)
        """
        if self.kernel == "rbf":
            if self.kernel_width is not None:
                kernel_width = self.kernel_width
            else:
                kernel_width = 1.0
            latent_dim = self.concept_reps.shape[-1]
            # We unstack the tensors to return a kernel matrix of shape len(h1) x len(h2)!
            return lambda h1, h2: torch.exp(
                -torch.sum(
                    ((h1.unsqueeze(1) - h2.unsqueeze(0)) / (latent_dim * kernel_width))
                    ** 2,
                    dim=-1,
                )
            )
        elif self.kernel == "linear":
            return lambda h1, h2: torch.einsum(
                "abi, abi -> ab", h1.unsqueeze(1), h2.unsqueeze(0)
            )

    def concept_density(
        self, latent_reps: torch.Tensor, positive_set: bool
    ) -> torch.Tensor:
        """
        Computes the concept density for the given latent representations
        Args:
            latent_reps: latent representations for which the concept density should be evaluated
            positive_set: if True, only compute the density for the positive set. If False, only for the negative.


        Returns:
            The density of the latent representations under the relevant concept set
        """
        kernel = self.get_kernel_function()
        latent_reps = latent_reps.to(self.device)
        concept_reps = torch.from_numpy(self.get_concept_reps(positive_set)).to(
            self.device
        )
        density = kernel(concept_reps, latent_reps).mean(dim=0)
        return density

    def tune_kernel_width(self, concept_reps: np.ndarray, concept_labels: np.ndarray):
        """
        Args:
            concept_reps: training representations
            concept_labels: training labels
        Tune the kernel width to achieve good training classification accuracy with a Parzen classifier
        Returns:

        """
        super(CAR, self).fit(concept_reps, concept_labels)

        def train_acc(trial):
            kernel_width = trial.suggest_float("kernel_width", 0.1, 50)
            self.kernel_width = kernel_width
            density = []
            for reps_batch in np.split(concept_reps, self.batch_size):
                density.append(
                    self.concept_importance(torch.from_numpy(reps_batch)).cpu().numpy()
                )
            density = np.concatenate(density)
            return accuracy_score((density > 0).astype(int), concept_labels)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(train_acc, n_trials=1000)
        self.kernel_width = study.best_params["kernel_width"]
        logging.info(
            f"Optimal kernel width {self.kernel_width:.3g} with training accuracy {study.best_value:.2g}"
        )

    def fit_cv(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels) by tuning the kernel width
        Args:
            concept_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(CAR, self).fit(concept_reps, concept_labels)

        X_train, X_val, y_train, y_val = train_test_split(
            concept_reps,
            concept_labels,
            test_size=int(0.3 * len(concept_reps)),
            stratify=concept_labels,
        )

        def objective(trial: optuna.Trial) -> float:
            kernel = trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            )
            gamma = trial.suggest_loguniform("gamma", 1e-3, 1e3)
            C = trial.suggest_loguniform("C", 1e-3, 1e3)
            classifier = SVC(kernel=kernel, gamma=gamma, C=C)
            classifier.fit(X_train, y_train)
            return accuracy_score(y_val, classifier.predict(X_val))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200, show_progress_bar=True)
        best_params = study.best_params
        self.classifier = SVC(**best_params)
        self.classifier.fit(concept_reps, concept_labels)
        self.kernel_width = best_params["gamma"]
        logging.info(
            f"Optimal hyperparameters {best_params} with validation accuracy {study.best_value:.2g}"
        )

    def concept_sensitivity_importance(
        self,
        latent_reps: np.ndarray,
        labels: torch.Tensor = None,
        num_classes: int = None,
        rep_to_output: Callable = None,
    ) -> np.ndarray:
        """
        Compute the concept sensitivity of a set of predictions
        Args:
            latent_reps: representations of the test examples
            labels: the labels associated to the representations one-hot encoded
            num_classes: the number of classes
            rep_to_output: black-box mapping the representation space to the output space
        Returns:
            concepts scores for each example
        """
        one_hot_labels = F.one_hot(labels, num_classes).to(self.device)
        latent_reps = torch.from_numpy(latent_reps).to(self.device).requires_grad_()
        outputs = rep_to_output(latent_reps)
        grads = torch.autograd.grad(outputs, latent_reps, grad_outputs=one_hot_labels)[
            0
        ]

        densities = self.concept_importance(latent_reps).view((-1, 1))
        cavs = torch.autograd.grad(
            densities,
            latent_reps,
            grad_outputs=torch.ones((len(densities), 1)).to(self.device),
        )[0]

        if len(grads.shape) > 2:
            grads = grads.flatten(start_dim=1)
        if len(cavs.shape) > 2:
            cavs = cavs.flatten(start_dim=1)
        return torch.einsum("bi,bi->b", cavs, grads).detach().cpu().numpy()
