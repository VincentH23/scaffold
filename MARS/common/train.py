from torch.utils.data import DataLoader
from ..datasets.datasets import ImitationDataset
import dgl
def imitation_training(agent, epochs, dataset, optimizer):
    n_nodes = dgl.batch(dataset.graphs).number_of_nodes()
    avg_size = int(n_nodes/len(dataset))
    batch_size = int(32*20 /avg_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn= ImitationDataset.collate_fn)

    for epoch in range(epochs) :
        for batch in loader:
            agent.train()
            optimizer.zero_grad()
            batch_size, metric_values = agent.global_act_loss(batch)
            loss = metric_values[0]
            loss.backward()
            optimizer.step()
