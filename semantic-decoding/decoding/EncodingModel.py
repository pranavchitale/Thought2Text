import numpy as np
import torch
from train_MLP import MLP
import config

torch.set_default_tensor_type(torch.FloatTensor)

class EncodingModel():
    """class for computing the likelihood of observing brain recordings given a word sequence
    """
    def __init__(self, resp, weights, voxels, sigma, mlp_path=None, device = "cpu"):
        self.device = device
        self.sigma = sigma
        self.resp = torch.from_numpy(resp[:, voxels]).float().to(self.device)
        if mlp_path:
            self.model = MLP(3072, config.VOXELS).to(self.device)
            model_state_dict = torch.load(mlp_path, map_location=self.device)
            if "module." in list(model_state_dict.keys())[0]:
                model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
            self.model.load_state_dict(model_state_dict)
            self.variant = "mlp"
        else:
            self.weights = torch.from_numpy(weights[:, voxels]).float().to(self.device)
            self.variant = "base"
        
    def set_shrinkage(self, alpha):
        """compute precision from empirical covariance with shrinkage factor alpha
        """
        precision = np.linalg.inv(self.sigma * (1 - alpha) + np.eye(len(self.sigma)) * alpha)
        self.precision = torch.from_numpy(precision).float().to(self.device)

    def prs(self, stim, trs):
        """compute P(R | S) on affected TRs for each hypothesis
        """
        with torch.no_grad():
            stim = stim.float().to(self.device)
            if self.variant == "base":
                presp = torch.matmul(stim, self.weights)
            else:
                presp = self.model(stim)
            diff = presp - self.resp[trs] # encoding model residuals
            multi = torch.matmul(torch.matmul(diff, self.precision), diff.permute(0, 2, 1))
            return -0.5 * multi.diagonal(dim1 = -2, dim2 = -1).sum(dim = 1).detach().cpu().numpy()
    