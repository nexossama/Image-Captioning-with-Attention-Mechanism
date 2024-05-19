from collections import Counter
import os
from typing import List
import spacy
import torch
from torch.utils.data import DataLoader,Dataset
from PIL import Image

class Vocabulary:
    #using spacy for text tokenization 
    spacy_eng = spacy.load("en_core_web_sm")

    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.index_to_str = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        self.str_to_index = {v:k for k,v in self.index_to_str.items()}
        
        #threshold for unkown words
        self.freq_threshold = freq_threshold
        
    def __len__(self): 
        return len(self.index_to_str)
    
    def build_vocab(self, sentence_list:List):
        tocken_frequencies = Counter()
        tocken_index = 4 # because 0->3 are already defined
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                tocken_frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if tocken_frequencies[word] == self.freq_threshold:
                    self.str_to_index[word] = tocken_index
                    self.index_to_str[tocken_index] = word
                    tocken_index += 1
    
    def sentence_to_indices(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.str_to_index[token] if token in self.str_to_index else self.str_to_index["<UNK>"] for token in tokenized_text ]  
    
    def indices_to_sentence(indices, index_to_str):
        """ Convert a list of indices back to a sentence using a provided index_to_str mapping """
        return ' '.join([index_to_str[index] for index in indices])
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in Vocabulary.spacy_eng.tokenizer(text)]

class FlickrDataset(Dataset):
    """
    FlickrDataset object for easy manipulation
    """
    def __init__(self,images_dir,captions_df,transform=None,freq_threshold=5):
        self.images_dir = images_dir #root directory of flickr that contains the images folder
        self.captions_df = captions_df
        self.transform = transform
        
        #Get image and caption colum from the dataframe
        self.imgs = self.captions_df["image"]
        self.captions = self.captions_df["caption"]
        
        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self,index):
        caption = self.captions[index]
        img_name = self.imgs[index]
        img_location = os.path.join(self.images_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #sentence_to_indices the caption text
        caption_vec = []
        caption_vec += [self.vocab.str_to_index["<SOS>"]]
        caption_vec += self.vocab.sentence_to_indices(caption)
        caption_vec += [self.vocab.str_to_index["<EOS>"]]
        
        return img, torch.tensor(caption_vec)
