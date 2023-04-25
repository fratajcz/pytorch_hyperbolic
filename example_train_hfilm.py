from torch_hyperbolic.models.hgnn_film import HFiLM
from torch_geometric.datasets import Planetoid
import torch

torch.set_default_dtype(torch.float64)

dataset = Planetoid(root='/tmp/Cora', name='Cora')

input_dim = dataset.num_node_features

output_dim = dataset.num_classes
hidden_dim = 12
loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
model = HFiLM(in_channels=input_dim, out_channels=output_dim, num_relations=2, hidden_dim=hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

y_onehot = torch.FloatTensor(dataset.y.shape[0], output_dim)
y_onehot.zero_()
y_onehot.scatter_(1, dataset.y.unsqueeze(-1), 1)

edge_index = torch.cat((dataset.edge_index, dataset.edge_index), dim=1)
edge_type = torch.cat((torch.zeros_like(dataset.edge_index)[0, :], torch.ones_like(dataset.edge_index)[0, :]), dim=0)
epochs = 100
print("Model curvatures before training: {}".format([round(x.detach().item(), 3) for x in model.curvatures]))
for epoch in range(epochs):
    # train
    model.zero_grad()
    out = model(dataset.x, edge_index, edge_type)
    flag = torch.any(torch.isnan(out))
    loss = loss_function(out[dataset.train_mask].squeeze(), y_onehot[dataset.train_mask].squeeze()).squeeze()
    flag = torch.any(torch.isnan(loss))
    loss = loss.mean()
    loss.backward()
    optimizer.step()

    # val
    with torch.no_grad():
        val_loss = loss_function(out[dataset.val_mask].squeeze(), y_onehot[dataset.val_mask].squeeze()).squeeze().mean()

    if (epoch + 1) % 5 == 0:
        print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch + 1, loss, val_loss))

# test
with torch.no_grad():
    test_loss = loss_function(out[dataset.test_mask].squeeze(), y_onehot[dataset.test_mask].squeeze()).squeeze().mean()
    print("Test Loss: {:.2f}".format(test_loss))
    print("Model curvatures after training: {}".format([round(x.detach().item(), 3) for x in model.curvatures]))