import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Neural Network is a Module. 
# It is also a composition of other modules. 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # a module that flatten an input
        self.flatten = nn.Flatten()

        # a sequential module composes of smaller modules
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print(model)

    # after initialize the network module 
    # the network object contains paramerters
    #print(model.parameters)
    list_of_parameters = model.parameters()
    for param in list_of_parameters:
        print(param.shape)


    x = torch.rand(1, 28, 28, device=device)
    y_pred_logits = model(x) # unnormalized output
    print(y_pred_logits)

    y_target = torch.tensor([[1,0,0,0,0,0,0,0,0,0]], device=device).float() #[1, 10]

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred_logits, y_target)
    print(loss)

    # automatic gradient computation
    loss.backward()
    for param in model.parameters():
        print(param.grad.shape) #dL/dw


