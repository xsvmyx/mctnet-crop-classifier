from mctnet_submodules import CNNSubmodule,TransformerSubmodule
import torch 
import torch.nn as nn

class MCTNet(nn.Module):
    def __init__(self, num_classes, in_channels=10): 
        super(MCTNet, self).__init__()
        
        # Calcul automatique des canaux par stage
        c1 = in_channels
        c2 = in_channels * 2 # Car le cat(CNN, Trans) double la taille
        c3 = in_channels * 4 
        out_features = in_channels * 8 # La taille qui arrive au MLP
        
        # STAGE 1 (d_model = c1)
        self.cnn1 = CNNSubmodule(in_channels=c1)
        self.trans1 = TransformerSubmodule(d_model=c1, use_alpe=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # STAGE 2 (d_model = c2)
        self.cnn2 = CNNSubmodule(in_channels=c2)
        self.trans2 = TransformerSubmodule(d_model=c2, use_alpe=False)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # STAGE 3 (d_model = c3)
        self.cnn3 = CNNSubmodule(in_channels=c3)
        self.trans3 = TransformerSubmodule(d_model=c3, use_alpe=False)
        self.global_pool = nn.AdaptiveMaxPool1d(1) 
        
        
        # MLP Final sadapte automatiquement a 80 88 96
        self.mlp = nn.Sequential(
            nn.Linear(out_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask):
        # CTFusion 1
        c1 = self.cnn1(x)
        t1 = self.trans1(x, mask)
        x = torch.cat([c1, t1], dim=-1) # Double les canaux
        x = self.pool1(x.transpose(1, 2)).transpose(1, 2) 
        
        # CTFusion 2
        c2 = self.cnn2(x)
        t2 = self.trans2(x) 
        x = torch.cat([c2, t2], dim=-1) # Double les canaux
        x = self.pool2(x.transpose(1, 2)).transpose(1, 2) 
        
        # CTFusion 3
        c3 = self.cnn3(x)
        t3 = self.trans3(x)
        x = torch.cat([c3, t3], dim=-1) # Double les canaux
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1) 
        
        return self.mlp(x)
