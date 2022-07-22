import unittest
import mindspore
import torch
import numpy as np
import mindspore.nn as msnn
import torch.nn as ptnn
from src.grad import grad, value_and_grad

class NetMS(msnn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, has_bias=True):
        super().__init__()
        self.conv2d = msnn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                     pad_mode='valid', has_bias=has_bias)

    def construct(self, inputs):
        return self.conv2d(inputs)

class NetPT(ptnn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, has_bias=True):
        super().__init__()
        self.conv2d = ptnn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=has_bias)

    def forward(self, inputs):
        return self.conv2d(inputs)

class TestGrad(unittest.TestCase):
    def setUp(self):
        self.in_channels = 8
        self.out_channels = 16
        self.kernel_size = (2, 2)
        self.weight = np.random.randn(self.out_channels, self.in_channels, *self.kernel_size).astype(np.float32)
        self.bias = np.random.randn(self.out_channels).astype(np.float32)

        self.ms_net = NetMS(self.in_channels, self.out_channels, self.kernel_size)
        self.pt_net = NetPT(self.in_channels, self.out_channels, self.kernel_size)
        
        self.ms_net.conv2d.weight.set_data(mindspore.Tensor(self.weight))
        self.ms_net.conv2d.bias.set_data(mindspore.Tensor(self.bias))

        self.pt_net.conv2d.weight = ptnn.Parameter(torch.tensor(self.weight))
        self.pt_net.conv2d.bias = ptnn.Parameter(torch.tensor(self.bias))

        self.inputs = np.random.randn(8, self.in_channels, 32, 32).astype(np.float32)

    def test_forward(self):
        out_ms = self.ms_net(mindspore.Tensor(self.inputs))
        out_pt = self.pt_net(torch.tensor(self.inputs))

        assert np.allclose(out_ms.asnumpy(), out_pt.detach().numpy(), 1e-4, 1e-4)
    
    def test_backward(self):
        ms_net = self.ms_net
        
        def ms_forward(inputs):
            out = ms_net(inputs)
            loss = -out.mean()
        
            return loss
        
        ms_grad_fn = value_and_grad(ms_forward, self.ms_net.trainable_params())
        loss_ms, grads_ms = ms_grad_fn(mindspore.Tensor(self.inputs))

        def pt_forward(inputs):
            out = self.pt_net(inputs)
            loss = -out.mean()
            return loss
        
        loss_pt = pt_forward(torch.tensor(self.inputs))
        loss_pt.backward()

        assert np.allclose(loss_ms.asnumpy(), loss_pt.detach().numpy(), 1e-4, 1e-4)

        pt_params = [param for param in self.pt_net.parameters()]
        assert len(pt_params) == len(grads_ms)

        for grad_ms, param_pt in zip(grads_ms, pt_params):
            assert np.allclose(grad_ms.asnumpy(), param_pt.grad.detach().numpy(), 1e-4, 1e-4)

    def test_gradient_panelty(self):
        ms_grad_fn = grad(self.ms_net)

        def pt_grad_fn(inputs):
            inputs = torch.autograd.Variable(inputs, requires_grad=True)
            outs = self.pt_net(inputs)
            gradients = torch.autograd.grad(outputs=outs, inputs=inputs,
                                            grad_outputs=torch.ones(outs.size()),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            return gradients
        
        pt_grads = pt_grad_fn(torch.tensor(self.inputs))
        ms_grads = ms_grad_fn(mindspore.Tensor(self.inputs))[0]
        
        assert np.allclose(ms_grads.asnumpy(), pt_grads.detach().numpy(), 1e-4, 1e-4)


        ms_net = self.ms_net
        
        def ms_forward(inputs):
            out = ms_net(inputs)
            loss = -out.mean() + ms_grad_fn(inputs)[0].mean()

            return loss
        
        ms_fn = value_and_grad(ms_forward, self.ms_net.trainable_params())
        loss_ms, grads_ms = ms_fn(mindspore.Tensor(self.inputs))

        def pt_forward(inputs):
            out = self.pt_net(inputs)
            loss = -out.mean() + pt_grad_fn(inputs).mean()
            return loss
        
        loss_pt = pt_forward(torch.tensor(self.inputs))
        loss_pt.backward()

        assert np.allclose(loss_ms.asnumpy(), loss_pt.detach().numpy(), 1e-3, 1e-3)

        pt_params = [param for param in self.pt_net.parameters()]
        assert len(pt_params) == len(grads_ms)

        for grad_ms, param_pt in zip(grads_ms, pt_params):
            assert np.allclose(grad_ms.asnumpy(), param_pt.grad.detach().numpy(), 1e-4, 1e-4)
