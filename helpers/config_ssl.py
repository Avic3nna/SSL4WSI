from torch.nn import L1Loss



class SSLModel():
    def __init__(self) -> None:
        super().__init__()

    def set_config():
        # Training Config
        # Define Network ViT backbone & Loss & Optimizer
        device = torch.device("cuda:0")
        model = ViTAutoEnc(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                pos_embed="conv",
                hidden_size=768,
                mlp_dim=3072,
        )

        model = model.to(device)

        # Define Hyper-paramters for training loop
        max_epochs = 500
        val_interval = 2
        batch_size = 4
        lr = 1e-4
        epoch_loss_values = []
        step_loss_values = []
        epoch_cl_loss_values = []
        epoch_recon_loss_values = []
        val_loss_values = []
        best_val_loss = 1000.0

        recon_loss = L1Loss()
        contrastive_loss = ContrastiveLoss(temperature=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
