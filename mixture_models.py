import numpy as np
import torch
import torch.nn as nn
from dataset import EuropeDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize_tensor(tensor, d):
    """
    Normalize the input tensor along the specified axis to have a mean of 0 and a std of 1.
    
    Parameters:
        tensor (torch.Tensor): Input tensor to normalize.
        d (int): Axis along which to normalize.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = torch.mean(tensor, dim=d, keepdim=True)
    std = torch.std(tensor, dim=d, keepdim=True)
    normalized = (tensor - mean) / std
    return normalized


class GMM(nn.Module):
    def __init__(self, n_components):
        """
        Gaussian Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of Gaussian components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Means of the Gaussian components (n_components x 2 for 2D data)
        self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of the variance of the Gaussian components (n_components x 2 for 2D data)
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # Log-variances (diagonal covariance)




    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """        
        #### YOUR CODE GOES HERE ####
        log_weights = torch.log_softmax(self.weights, dim=0)
        log_likelihoods = []
        for k in range(self.n_components):
            mean = self.means[k]
            log_var = self.log_variances[k]
            var = torch.exp(log_var)
            diff = X - mean
            exponent = -0.5 * torch.sum(diff * diff / var, dim=1)
            norm = -0.5 * torch.sum(log_var) - torch.log(torch.tensor(2 * torch.pi))
            log_likelihood = norm + exponent
            log_likelihoods.append(log_likelihood)
        log_likelihoods = torch.stack(log_likelihoods, dim=1)
        log_likelihood = torch.logsumexp(log_weights + log_likelihoods, dim=1)
        return log_likelihood

    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        #### YOUR CODE GOES HERE ####
        return -torch.mean(log_likelihood)


    def sample(self, n_samples):
        """
        Generate samples from the GMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        weights = torch.softmax(self.weights, dim=0)
        indices = torch.multinomial(weights, n_samples, replacement=True)
        samples = []
        for idx in indices:
            mean = self.means[idx]
            log_var = self.log_variances[idx]
            var = torch.exp(log_var)
            sample = torch.normal(mean, torch.sqrt(var))
            samples.append(sample)
        return torch.stack(samples).detach()
    
    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        mean = self.means[label]
        log_var = self.log_variances[label]
        var = torch.exp(log_var)
        std = torch.sqrt(var)
        z = torch.randn((n_samples, 2), device=mean.device)
        samples = mean + z * std
        return samples.detach()

class UMM(nn.Module):
    def __init__(self, n_components):
        """
        Uniform Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of uniform components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Center value of the uniform components (n_components x 2 for 2D data)
        self.centers = nn.Parameter(torch.randn(n_components, 2))

        # Log of size of the uniform components (n_components x 2 for 2D data)
        self.log_sizes = nn.Parameter(torch.log(3 * torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))


    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """
        #### YOUR CODE GOES HERE ####
        # (log pi_k)
        log_weights = torch.log_softmax(self.weights, dim=0)  # (n_components,)

        # exp(log_sizes)
        sizes = torch.exp(self.log_sizes)

        # Compute lower and upper bounds for each component
        lower = self.centers - sizes / 2
        upper = self.centers + sizes / 2

        # Expand dimensions for broadcasting: (n_samples, 1, 2) vs (1, n_components, 2)
        X_exp = X.unsqueeze(1)  # (n_samples, 1, 2)
        lower_exp = lower.unsqueeze(0)  # (1, n_components, 2)
        upper_exp = upper.unsqueeze(0)  # (1, n_components, 2)

        # Check if samples lie within the bounds for all dimensions
        within_bounds = (X_exp >= lower_exp) & (X_exp <= upper_exp)
        within_bounds = within_bounds.all(dim=-1)  # (n_samples, n_components)

        # Compute log p(x | k) = -log(s1) - log(s2) if within bounds, else -1e6
        sum_log_sizes = self.log_sizes.sum(dim=-1)  # (n_components,)
        log_pxk = torch.where(
            within_bounds,
            -sum_log_sizes,  # -log(s1) - log(s2)
            torch.tensor(-1e6, device=X.device)
        )

        log_likelihood = torch.logsumexp(log_weights + log_pxk, dim=1)  # (n_samples,)

        return log_likelihood
    
    
    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        #### YOUR CODE GOES HERE ####
        return -torch.mean(log_likelihood)

    def sample(self, n_samples):
        """
        Generate samples from the UMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        weights = torch.softmax(self.weights, dim=0)
        indices = torch.multinomial(weights, n_samples, replacement=True)

        samples = []
        for idx in indices:
            center = self.centers[idx]
            size = torch.exp(self.log_sizes[idx])

            # Calculate bounds
            lower = center - size / 2
            upper = center + size / 2

            # Use torch.distributions.Uniform
            uniform_dist = torch.distributions.Uniform(lower, upper)
            sample = uniform_dist.sample()
            samples.append(sample)

        return torch.stack(samples).detach()

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        center = self.centers[label]
        size = torch.exp(self.log_sizes[label])

        # Calculate bounds
        lower = center - (size / 2)
        upper = center + (size / 2)

        # Use torch.distributions.Uniform
        uniform_dist = torch.distributions.Uniform(lower, upper)
        samples = uniform_dist.sample((n_samples,))
        return samples.detach()

def train_model(model, model_name, num_epochs, train_loader, test_loader, show_graphs=False, which_epoch=[]):
    optimizer = torch.optim.Adam(model.parameters(), lr=GMM_lr)

    train_mean_log_likelihoods = []
    test_mean_log_likelihoods = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        running_log_likelihood = 0
        for batch in train_loader:
            features, _ = batch
            log_likelihood = model(features)
            loss = model.loss_function(log_likelihood)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_log_likelihood += torch.mean(log_likelihood).item()

        # Compute mean log-likelihood for training data
        train_mean_log_likelihood = running_log_likelihood / len(train_loader)
        train_mean_log_likelihoods.append(train_mean_log_likelihood)

        # Testing phase
        model.eval()
        running_log_likelihood = 0
        with torch.no_grad():
            for batch in test_loader:
                features, _ = batch
                log_likelihood = model(features)
                running_log_likelihood += torch.mean(log_likelihood).item()

        # Compute mean log-likelihood for testing data
        test_mean_log_likelihood = running_log_likelihood / len(test_loader)
        test_mean_log_likelihoods.append(test_mean_log_likelihood)

        print(
            f"Epoch {epoch + 1}, Train Log-Likelihood: {train_mean_log_likelihood}, Test Log-Likelihood: {test_mean_log_likelihood}")

        if show_graphs:
            if epoch + 1 in which_epoch:
                samples = model.sample(1000)
                plot_samples(samples, f"{model_name} n classes \nepoch: {epoch + 1}")
                plot_comp_samples(model, f"{model_name} n classes \n epoch: {epoch + 1}", model.n_components)

    return model, train_mean_log_likelihoods, test_mean_log_likelihoods


def plot_samples(samples, title):
    print(f"plotting samples for {title}")
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.title(title)
    plt.show()


def plot_comp_samples(model, title, n_comp):
    print(f"Plotting samples for {title}")
    plt.figure()
    twilight = plt.get_cmap('twilight')
    colors = twilight(np.linspace(0, 1.0, len(range(n_comp))+1))

    for comp in range(n_comp):
        samples = model.conditional_sample(100, comp)
        samples = samples.detach().cpu().numpy()
        plt.scatter(
            samples[:, 0],
            samples[:, 1],
            c=[colors[comp]] * 100,
            label=f"Component {comp}",
            alpha=0.6
        )

    plt.title(title)
    plt.show()


def initialize_means(gmm, train_dataset):
    all_labels = sorted(torch.unique(train_dataset.labels).tolist())
    means = []
    for label in all_labels:
        label_features = train_dataset.get_features(label)
        mean = torch.mean(label_features, dim=0)
        means.append(mean)
    gmm.means.data = torch.stack(means)
    return gmm

def initialize_centers(umm, train_dataset):
    """
    Initialize the centers of UMM components using country means, similar to GMM
    """
    all_labels = sorted(torch.unique(train_dataset.labels).tolist())
    centers = []
    for label in all_labels:
        label_features = train_dataset.get_features(label)
        center = torch.mean(label_features, dim=0)
        centers.append(center)
    umm.centers.data = torch.stack(centers)
    return umm

def model_n_classes(model, model_name, num_epochs, train_loader, test_loader, init_means=False):
    print(f"{model_name} for {model.n_components} components")
    if model_name == "GMM":
        title = "GMM n classes"
    else:
        title = "UMM n classes"

    if init_means:
        if model_name == "GMM":
            model = initialize_means(model, train_dataset)
        else:
            model = initialize_centers(umm, train_dataset)
        title += ", initialized means"

    model, train_mean_log_likelihoods, test_mean_log_likelihoods = train_model(
        model, model_name, num_epochs, train_loader,
        test_loader,
        show_graphs=True,
        which_epoch=[1, 10, 20, 30, 40, 50])

    plt.figure()
    plt.plot(train_mean_log_likelihoods, label='Train Log-Likelihood')
    plt.plot(test_mean_log_likelihoods, label='Test Log-Likelihood')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Log-Likelihood')
    plt.title(title + '\nMean Log-Likelihood vs. Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    torch.manual_seed(42)
    np.random.seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    batch_size = 4096
    num_epochs = 50
    # Use Adam optimizer
    #TODO: Determine learning rate
    # learning_rate for GMM = 0.01
    # learning_rate for UMM = 0.001

    GMM_lr = 0.01
    UMM_lr = 0.001

    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #### YOUR CODE GOES HERE ####

    n_classes = len(torch.unique(train_dataset.labels))

    # print("\nStarting GMM Experiments...")
    # for n_comp in [1, 5, 10]:
    #     print(f"GMM: {n_comp}...")
    #     gmm = GMM(n_components=n_comp)
    #     model, _, _ = train_model(gmm, "GMM", num_epochs, train_loader, test_loader)
    #
    #     samples = model.sample(1000)
    #     plot_samples(samples, f"1000 samples from GMM for n_components: {n_comp}")
    #
    #     plot_comp_samples(model, f"100 samples from GMM for each component in : {n_comp}", n_comp)
    #
    # gmm = GMM(n_components=n_classes)
    # model_n_classes(gmm, "GMM", num_epochs, train_loader, test_loader)
    #
    # gmm = GMM(n_components=n_classes)
    # model(gmm, "GMM", num_epochs, train_loader, test_loader, init_means=True)

    print("\nStarting UMM Experiments...")
    for n_comp in [1, 5, 10]:
        print(f"UMM: {n_comp}...")
        umm = UMM(n_components=n_comp)
        model, _, _ = train_model(umm, "UMM", num_epochs, train_loader, test_loader)

        samples = model.sample(1000)
        plot_samples(samples, f"1000 samples from UMM for n_components: {n_comp}")

        plot_comp_samples(model, f"100 samples from UMM for each component in : {n_comp}", n_comp)

    umm = UMM(n_components=n_classes)
    model_n_classes(umm, "UMM", num_epochs, train_loader, test_loader)

    umm = UMM(n_components=n_classes)
    model_n_classes(umm, "UMM", num_epochs, train_loader, test_loader, init_means=True)