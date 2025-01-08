#imports 
import os
import numpy as np
from collections import Counter
import spacy
# import pandas as pd
from PIL import Image
import typing
from typing import List
import dill
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
from torch import optim
from torch.nn import functional
from torchvision import models
from models.FlickrDataset import FlickrDataset


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features

#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.decoder_attention = nn.Linear(decoder_dim,attention_dim)
        self.encoder_attention = nn.Linear(encoder_dim,attention_dim)
        self.cell_attention = nn.Linear(decoder_dim, attention_dim)

        self.attention_layer = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state, cell_state):
        encoder_attention_states = self.encoder_attention(features)     #(batch_size,num_layers,attention_dim)
        decoder_attention_states = self.decoder_attention(hidden_state) #(batch_size,attention_dim)
        cell_attention_states = self.cell_attention(cell_state)         #(batch_size, attention_dim)

        combined_states = torch.tanh(encoder_attention_states + decoder_attention_states.unsqueeze(1) + cell_attention_states.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.attention_layer(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        
        alpha = functional.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha,attention_weights
        
#Attention Decoder
class DecoderLSTM(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        
        self.logits_layer = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def forward(self, features, captions):
        
        #vectorize the caption
        embeds = self.embedding(captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(self.device)
                
        for s in range(seq_length):
            alpha,context = self.attention(features, h,c)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.logits_layer(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas
    
    def generate_caption(self, features, max_len=20, vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # Starting input
        word = torch.tensor(vocab.str_to_index['<SOS>']).view(1, -1).to(self.device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h,c)

            # Store the alpha score
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.logits_layer(self.drop(h))
            output = output.view(batch_size, -1)

            # Replace <UNK> token with the most probable word
            predicted_word_idx = output.argmax(dim=1)
            if predicted_word_idx.item() == vocab.str_to_index['<UNK>']:
                _, next_highest = torch.topk(output, 2, dim=1)
                predicted_word_idx = next_highest[:, 1]  # Select the second most probable word
            # Save the generated word
            captions.append(predicted_word_idx.item())

            # End if <EOS detected>
            if vocab.index_to_str[predicted_word_idx.item()] == "<EOS>":
                break

            # Send the generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        # Convert the vocab idx to words and return sentence
        return [vocab.index_to_str[idx] for idx in captions], alphas
    
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,image_transformation,drop_prob=0.3):
        super().__init__()
        self.embed_size=embed_size
        self.vocab_size=vocab_size
        self.attention_dim=attention_dim
        self.encoder_dim=encoder_dim
        self.decoder_dim=decoder_dim
        
        self.encoder = EncoderCNN()
        self.decoder = DecoderLSTM(
            embed_size=self.embed_size,
            vocab_size = self.vocab_size,
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim
        )
        self.image_transformation=image_transformation
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_dataset=None

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    #generate caption
    def get_caps_from_image(self,image,show_image=False):
        features_tensors = self.image_transformation(image).unsqueeze(0)
        #generate the caption
        # self.eval()
        with torch.no_grad():
            features = self.encoder(features_tensors.to(self.device))
            
            src_path=os.path.dirname(os.path.dirname(__file__))

            caps,alphas = self.decoder.generate_caption(features,vocab=self.train_dataset.vocab)
            caption = ' '.join(caps)
            if show_image:
                show_image(features_tensors[0],title=caption)
        return caption,caps,alphas
        
        
    def save_model(self,num_epochs):
        model_state = {
            'num_epochs':num_epochs,
            'embed_size':self.embed_size,
            'vocab_size':self.vocab_size,
            'attention_dim':self.attention_dim,
            'encoder_dim':self.encoder_dim,
            'decoder_dim':self.decoder_dim,
            'image_transformation':self.image_transformation,
            'state_dict':self.state_dict()
        }

        torch.save(model_state,f'attention_model_state_{num_epochs}.pth')
    
    @staticmethod
    def load_model(path):
        # Define the path to your saved model state dictionary
        model_state_path = path

        # Load the state dictionary
        model_state = torch.load(model_state_path,map_location=torch.device('cpu'))

        # Initialize your model architecture
        model = EncoderDecoder(
            embed_size=model_state['embed_size'],
            vocab_size=model_state['vocab_size'],
            attention_dim=model_state['attention_dim'],
            encoder_dim=model_state['encoder_dim'],
            decoder_dim=model_state['decoder_dim'],
            image_transformation=model_state['image_transformation']
        )

        # Load the state dictionary into the model
        model.load_state_dict(model_state['state_dict'])

        # If the model was trained on GPU and you want to use it on GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Ensure the model is in evaluation mode
        model.eval()
        return model


