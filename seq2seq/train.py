from dataset import train_iter, collate_fn, pad_idx
from torch.utils.data import DataLoader
from torch.utils.data.backward_compatibility import worker_init_fn


def main():
    
    # 1. get dataloader
    train_dataloader = DataLoader(train_iter, 
                        batch_size=6, 
                        collate_fn=collate_fn,
                        num_workers=2,
                        worker_init_fn=worker_init_fn,
                        drop_last=True)

    # 2. model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_net = RnnEncoder().to(device)
    decoder_net = RnnDecoder().to(device)
    model = Seq2Seq(encoder_net, 
                          decoder_net).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = opim.Adam(model.parameters(), lr=0.0001)

    # 3. logger
    logger = Logger()

    # 4. training loop
    