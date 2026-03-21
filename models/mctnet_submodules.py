import torch
import torch.nn as nn
import math

class CNNSubmodule(nn.Module):
    def __init__(self, in_channels=10): 
        super(CNNSubmodule, self).__init__()
    
        
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
       
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [Batch, 36, 10]
        x = x.transpose(1, 2) # [Batch, 10, 36]
        identity = x 
 
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity 
        
        out = self.relu(out)
        
        return out.transpose(1, 2) # Retour en [Batch, 36, 10]
    





class ALPE(nn.Module):
    def __init__(self, d_model=10, max_len=36):
        super(ALPE, self).__init__()
        self.d_model = d_model
        
        #PE(t) = Initial Absolute Positional Encoding
        pe = torch.zeros(max_len, d_model)

        # cette liste sert a calculer les sinus et les cosinus car il faut un angle. Cet angle dépend du jour où on se trouve.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, 36, d_model]

        #Conv1D Layer
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        #ECA Module
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()



    def eca(self, x):
        y = self.avg_pool(x) 
        y = self.eca_conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y) 

        return x * y.expand_as(x)



    def forward(self, x, input2_mask):
        # x: [Batch, 36, 10] (Input 1 - Bandes spectrales)
        # input2_mask: [Batch, 36, 1] (Input 2 - Masque de nuages)
        
        b, t, c = x.size()
        
        #on recup la matrice de position initiale pour chaque echantillon
        pos_vector = self.pe[:, :t, :].clone().repeat(b, 1, 1) # [Batch, 36, 10]
        
        #si le jour est nuageux , on met tout son embedding à 0 pour que le modèle ne puisse pas s'appuyer dessus
        pos_vector = pos_vector * input2_mask 
        
        # format attendu par pythorch Shape: [Batch, 10, 36]
        pos_vector = pos_vector.transpose(1, 2)
        
        # le conv1d va apprendre à ajuster les embeddings positionnels en fonction des données d'entrée, en tenant compte des jours nuageux.
        pos_vector = self.conv1d(pos_vector)
    
        pos_vector = self.eca(pos_vector)
        
        
        return pos_vector.transpose(1, 2) # Sortie: [Batch, 36, 10]
    





class TransformerSubmodule(nn.Module):
    def __init__(self, d_model=10, nhead=2, dim_feedforward=64, use_alpe=False):
        super(TransformerSubmodule, self).__init__()
        self.use_alpe = use_alpe
        
        if self.use_alpe:    
            self.alpe = ALPE(d_model=d_model)
    
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, input2_mask=None):
        # x: Input spectral [Batch, 36, 10]
        # input2_mask: Le masque pour ALPE
        

        #stage 1 uniquement
        if self.use_alpe and input2_mask is not None:
                
                pos_info = self.alpe(x, input2_mask)
                x = x + pos_info
        
        
        # Ici se passent le Multi-Head Attention, Add & Norm, Feed Forward
        out = self.transformer_encoder(x)
        
        return out # Sortie [Batch, 36, 10]