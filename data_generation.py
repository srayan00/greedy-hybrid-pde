import numpy as np
import matplotlib.pyplot as plt
import torch


class GaussianRandomField:

    def __init__(self, num_samples, dim, alpha, beta, gamma, device, seed=None):
        if num_samples % 2 == 0:
            raise ValueError("Number of samples must be odd")
        self.k_max = num_samples // 2
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device if device else torch.cpu()
        self.num_samples = num_samples


        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if dim == 1:
            k_range = torch.arange(0, self.k_max + 1).to(self.device)
            k_range_negative = torch.arange(-self.k_max, 0).to(self.device)
            self.krange = torch.concatenate((k_range, k_range_negative))
            self.psd = self._compute_psd_1d(self.krange)
        elif dim == 2:
            kx = torch.arange(0, self.k_max + 1).to(self.device)
            k_range_negative = torch.arange(-self.k_max, 0).to(self.device)
            kx = torch.concatenate((kx, k_range_negative))
            ky = torch.arange(0, self.k_max + 1).to(self.device)
            self.kx, self.ky = torch.meshgrid(kx, ky, indexing = "ij")
            self.psd = self._compute_psd_2d(self.kx, self.ky)
        else:
            raise ValueError("Dimension must be either 1 or 2.")

    def _compute_psd_1d(self, krange):
        pi = torch.tensor(np.pi).to(self.device)
        psd = (self.alpha ** (1/2)) * (4 * pi**2 * krange**2 + self.beta)**(-self.gamma/2)
        return psd
    
    def _compute_psd_2d(self, kx, ky):
        pi = torch.tensor(np.pi).to(self.device)
        psd = (self.alpha ** (1/2)) * (4 * pi**2 * (kx**2 + ky**2) + self.beta)**(-self.gamma/2)
        return psd
    
    def generate(self, n_samples, pushfoward = torch.exp):
        if self.dim == 1:
            result = self._generate_1d(n_samples)
        elif self.dim == 2:
            result = self._generate_2d(n_samples)
        else:
            raise NotImplementedError("2D Gaussian random field generation is not implemented yet.")
        if pushfoward:
            result = pushfoward(result)
        return result
        
        
    
    def _generate_1d(self, n_samples):
        psd = self.psd
        real_positive_freq = torch.randn(n_samples, self.k_max + 1).to(self.device) # Real-valued part
        complex_positive_freq = torch.randn(n_samples, self.k_max + 1).to(self.device) # complex-valued part
        positive_freq = real_positive_freq + 1.j *complex_positive_freq # Zs for positive frequency
        conjugate_positive_freq = torch.conj(positive_freq) # conjugate frequencies
        freq = torch.concatenate((positive_freq, torch.flip(conjugate_positive_freq, dims = [1])[:, :-1]), axis = 1)  # f^*(k) = f(-k)
        freq[:, 0] = 0 # torch.randn(n_samples)
        fourier_coeffs = psd * freq
        field = torch.fft.ifft(fourier_coeffs, n= self.num_samples, norm = "ortho")
        return field.real
    
    def visualize(self):
        samples = self.generate(10)
        sharey = True if self.dim == 2 else False
        fig, ax = plt.subplots(2, 5, sharey = sharey)
        for i in range(2):
            for j in range(5):
                idx = 2*i + j
                curr = samples[idx]
                if self.dim == 1:
                    ax[i][j].plot(np.linspace(0, 1, self.num_samples), curr)
                else:
                    kx = np.linspace(0, 1, self.num_samples)
                    ky = np.linspace(0, 1, self.num_samples)
                    kx, ky = np.meshgrid(kx, ky)
                    z = ax[i][j].pcolormesh(kx, ky, curr)
                    plt.colorbar(z, ax=ax[i][j])
                ax[i][j].set_title(f"Sample {idx}")
        plt.show()
        



    # def _generate_1d(self, n_samples):
    #     zs = torch.random.normal(size = (n_samples, self.k_max + 1)) + 1.j * torch.random.normal(size = (n_samples, self.k_max + 1))
    #     zs[:, 0] = torch.randn(n_samples)
    #     coefs = self.psd * zs
    #     field = torch.fft.irfft(coefs, n = self.num_samples, norm= "ortho")

    def _generate_2d(self, n_samples):
        psd = self.psd
        real_positive_freq = torch.randn(n_samples, 2*self.k_max + 1, self.k_max + 1).to(self.device) # Real-valued part
        complex_positive_freq = torch.randn(n_samples, 2*self.k_max + 1, self.k_max + 1).to(self.device)
        hermitian_half = real_positive_freq + 1.j *complex_positive_freq
        hermitian_half[:, 0, 0] = 0 # torch.randn(n_samples)
        fourier_coefs = psd * hermitian_half
        return torch.fft.irfft2(fourier_coefs, s= (self.num_samples, self.num_samples), norm = "ortho")



class GaussianRandomFieldHierarchical:
    def __init__(self, num_samples, dim, alpha_min, alpha_max, beta_min, beta_max, gamma_list, device, seed=None):
        if num_samples % 2 == 0:
            raise ValueError("Number of samples must be odd")
        self.k_max = num_samples // 2
        self.dim = dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_list = gamma_list
        # self.alpha = alpha
        # self.beta = beta
        # self.gamma = gamma
        self.device = device if device else torch.cpu()
        self.num_samples = num_samples


        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if dim == 1:
            k_range = torch.arange(0, self.k_max + 1).to(self.device)
            k_range_negative = torch.arange(-self.k_max, 0).to(self.device)
            self.krange = torch.concatenate((k_range, k_range_negative))
            # self.psd = self._compute_psd_1d(self.krange)
        elif dim == 2:
            kx = torch.arange(0, self.k_max + 1).to(self.device)
            k_range_negative = torch.arange(-self.k_max, 0).to(self.device)
            kx = torch.concatenate((kx, k_range_negative))
            ky = torch.arange(0, self.k_max + 1).to(self.device)
            self.kx, self.ky = torch.meshgrid(kx, ky, indexing = "ij")
            # self.psd = self._compute_psd_2d(self.kx, self.ky)
        else:
            raise ValueError("Dimension must be either 1 or 2.")


    def _sample_log_uniform(self, n, a, b):
        a = torch.Tensor([a], device=self.device)
        b = torch.Tensor([b], device=self.device)
        u = torch.rand(n, device=self.device)
        return torch.exp(u*(torch.log(b) - torch.log(a)) + torch.log(a))
    
    def _sample_gamma(self, n, list_of_val):
        list_of_val = torch.Tensor(list_of_val, device=self.device)
        idxs = torch.randint(high = list_of_val.size(0), size = (n,), device =self.device)
        return list_of_val[idxs]

    def _compute_psd_1d(self, krange, n_samples):
        alpha = self._sample_log_uniform(n_samples, self.alpha_min, self.alpha_max)
        beta = self._sample_log_uniform(n_samples, self.beta_min, self.beta_max)
        gamma = self._sample_gamma(n_samples, self.gamma_list)
        pi = torch.tensor(np.pi).to(self.device)
        psd = (alpha[:, None] ** (1/2)) * (4 * pi**2 * krange**2 + beta[:, None])**(-gamma[:, None]/2)
        return psd
    
    def _compute_psd_2d(self, kx, ky, n_samples):
        alpha = self._sample_log_uniform(n_samples, self.alpha_min, self.alpha_max)
        beta = self._sample_log_uniform(n_samples, self.beta_min, self.beta_max)
        gamma = self._sample_gamma(n_samples, self.gamma_list)
        pi = torch.tensor(np.pi).to(self.device)
        psd = (alpha[:, None, None] ** (1/2)) * (4 * pi**2 * (kx**2 + ky**2) + beta[:, None, None])**(-gamma[:, None, None]/2)
        return psd
    
    def generate(self, n_samples, pushfoward = torch.exp):
        if self.dim == 1:
            result = self._generate_1d(n_samples)
        elif self.dim == 2:
            result = self._generate_2d(n_samples)
        else:
            raise NotImplementedError("2D Gaussian random field generation is not implemented yet.")
        if pushfoward:
            result = pushfoward(result)
        return result
        
        
    
    def _generate_1d(self, n_samples):
        psd = self._compute_psd_1d(self.krange, n_samples)
        real_positive_freq = torch.randn(n_samples, self.k_max + 1).to(self.device) # Real-valued part
        complex_positive_freq = torch.randn(n_samples, self.k_max + 1).to(self.device) # complex-valued part
        positive_freq = real_positive_freq + 1.j *complex_positive_freq # Zs for positive frequency
        conjugate_positive_freq = torch.conj(positive_freq) # conjugate frequencies
        freq = torch.concatenate((positive_freq, torch.flip(conjugate_positive_freq, dims = [1])[:, :-1]), axis = 1)  # f^*(k) = f(-k)
        freq[:, 0] = 0 # torch.randn(n_samples)
        fourier_coeffs = psd * freq
        field = torch.fft.ifft(fourier_coeffs, n= self.num_samples, norm = "ortho")
        return field.real
    
    def visualize(self):
        samples = self.generate(10)
        sharey = True if self.dim == 2 else False
        fig, ax = plt.subplots(2, 5, sharey = sharey)
        for i in range(2):
            for j in range(5):
                idx = 2*i + j
                curr = samples[idx]
                if self.dim == 1:
                    ax[i][j].plot(np.linspace(0, 1, self.num_samples), curr)
                else:
                    kx = np.linspace(0, 1, self.num_samples)
                    ky = np.linspace(0, 1, self.num_samples)
                    kx, ky = np.meshgrid(kx, ky)
                    z = ax[i][j].pcolormesh(kx, ky, curr)
                    plt.colorbar(z, ax=ax[i][j])
                ax[i][j].set_title(f"Sample {idx}")
        plt.show()
        



    # def _generate_1d(self, n_samples):
    #     zs = torch.random.normal(size = (n_samples, self.k_max + 1)) + 1.j * torch.random.normal(size = (n_samples, self.k_max + 1))
    #     zs[:, 0] = torch.randn(n_samples)
    #     coefs = self.psd * zs
    #     field = torch.fft.irfft(coefs, n = self.num_samples, norm= "ortho")

    def _generate_2d(self, n_samples):
        psd = self._compute_psd_2d(self.kx, self.ky, n_samples)
        real_positive_freq = torch.randn(n_samples, 2*self.k_max + 1, self.k_max + 1).to(self.device) # Real-valued part
        complex_positive_freq = torch.randn(n_samples, 2*self.k_max + 1, self.k_max + 1).to(self.device)
        hermitian_half = real_positive_freq + 1.j *complex_positive_freq
        hermitian_half[:, 0, 0] = 0 # torch.randn(n_samples)
        fourier_coefs = psd * hermitian_half
        return torch.fft.irfft2(fourier_coefs, s= (self.num_samples, self.num_samples), norm = "ortho")




    
class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input, u = self.data[idx]
        return input, u
