import torch

class KineticWrapper(torch.nn.Module):
    def __init__(self, model, first, second):
        super().__init__()
        self.model = model
        self.div_samples = 1
        self.first = first
        self.second = second

    def reset(self, logsig, time_length):
        self.model.reset(logsig, time_length)
        self.logsig_getter = self.model.logsig_getter

    def forward(self, t, z):

            with torch.set_grad_enabled(True):
                z1, z2 = z[0], z[1]
                if self.first:
                    z1.requires_grad_(True)
                if self.second:
                    z2.requires_grad_(True)
                for s_ in z[2:]:
                    s_.requires_grad_(True)
            
                dz1_dt, dz2_dt = self.model(t,(z1, z2))
                if self.first:
                    dz1_dt.requires_grad_(True)
                if self.second:
                    dz2_dt.requires_grad_(True)

                result = [dz1_dt, dz2_dt]
                if self.first:
                    z1_sqjacnorm = self.jacobian_frobenius_regularization_fn(z1, dz1_dt)
                    z1_quad = self.quadratic_cost(dz1_dt)
                    result.append(z1_sqjacnorm)
                    result.append(z1_quad)
                if self.second:
                    z2_sqjacnorm = self.jacobian_frobenius_regularization_fn(z2, dz2_dt)
                    z2_quad = self.quadratic_cost(dz2_dt)
                    result.append(z2_sqjacnorm)
                    result.append(z2_quad)
            # import pdb ; pdb.set_trace()
            return tuple(result)

    def jacobian_frobenius_regularization_fn(self, h0, dhdt):
        sqjacnorm = []
        for e in [torch.randn_like(h0) for k in range(self.div_samples)]:
            # import pdb ; pdb.set_trace()
            e_dhdt_dx = torch.autograd.grad(dhdt, h0, e, create_graph=True)[0]
            n = e_dhdt_dx.view(h0.size(0),-1).pow(2).mean(dim=1, keepdim=True)
            sqjacnorm.append(n)
        return torch.cat(sqjacnorm, dim=1).mean(dim=1)

    def quadratic_cost(self, dx):
        dx = dx.view(dx.shape[0], -1)
        return 0.5*dx.pow(2).mean(dim=-1)
