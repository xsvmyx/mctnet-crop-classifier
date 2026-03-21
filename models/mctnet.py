from mctnet_submodules import CNNSubmodule,TransformerSubmodule
import torch 
import torch.nn as nn
#ici on va assembler les sous modules pour faire le MCTNet
class MCTNet(nn.Module):
    def __init__(self, num_classes):
        super(MCTNet, self).__init__()
        
        # STAGE 1 (d_model=10, avec ALPE)
        self.cnn1 = CNNSubmodule(in_channels=10)
        self.trans1 = TransformerSubmodule(d_model=10, use_alpe=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # STAGE 2 (d_model=20, sans ALPE)
        self.cnn2 = CNNSubmodule(in_channels=20)
        self.trans2 = TransformerSubmodule(d_model=20, use_alpe=False)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # STAGE 3 (d_model=40, sans ALPE)
        self.cnn3 = CNNSubmodule(in_channels=40)
        self.trans3 = TransformerSubmodule(d_model=40, use_alpe=False)
        self.global_pool = nn.AdaptiveMaxPool1d(1) # Pour arriver à 80*1 peu importe la taille restante
        
        # MLP Final
        self.mlp = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask):
        #CTFusion 1
        c1 = self.cnn1(x)
        t1 = self.trans1(x, mask)
        x = torch.cat([c1, t1], dim=-1) # [B, 36, 20]
        x = self.pool1(x.transpose(1, 2)).transpose(1, 2) # [B, 18, 20]
        
        # CTFusion 2
        c2 = self.cnn2(x)
        t2 = self.trans2(x) 
        x = torch.cat([c2, t2], dim=-1) # [B, 18, 40]
        x = self.pool2(x.transpose(1, 2)).transpose(1, 2) # [B, 9, 40]
        
        # CTFusion 3
        c3 = self.cnn3(x)
        t3 = self.trans3(x)
        x = torch.cat([c3, t3], dim=-1) # [B, 9, 80]
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1) # [B, 80]
        
        return self.mlp(x)