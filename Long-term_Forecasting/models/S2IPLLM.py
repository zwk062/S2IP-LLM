#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr 
from .prompt import Prompt 






 





class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = 768
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
       

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        

       
        if configs.pretrained == True:
           
          
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

            
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())

        
        

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name  or 'wpe' in name:   #or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # False


       

        

        if self.task_name == 'long_term_forecast':
        
            self.in_layer = nn.Linear(configs.patch_size*3, configs.d_model)
            self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)) , configs.pred_len)
            
            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight)
                    
        
            
   
            
            for layer in (self.gpt2, self.in_layer, self.out_layer):       
                layer.cuda()
                layer.train()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):


        if self.task_name == 'long_term_forecast':
            dec_out,res = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :],res  # [B, L, D]
        
        
        return None

   

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        

        
         
            
        B, L, M = x_enc.shape # batchsize,sequence length,变量数
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
        torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # 把每个样本的每个变量展开为单独的序列，Transformer 是基于序列建模的，它最擅长处理 一维序列
        x = rearrange(x_enc, 'b l m -> (b m) l')   # [样本1-变量1的时间序列]

        # 时间序列分解
        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
            residuals = df - trend - seasonal
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined # (T, 3, M)
                
            

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy()) # (B, T, 3, M)
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c = 3) # 将每个成分（trend/seasonal/residual）分别视作一个长向量 (B, 3, M*T)
        x = self.padding_patch_layer(x) # 补0
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # 切分 Patch (B, 3, N, P) N: patch 数量 P: patch 大小
        x = rearrange(x, 'b c n p -> b n (c p)', c = 3)  # (B, N, 3 * P) 每个 patch 被拉成一个向量（包含 trend + seasonal + residual）
        pre_prompted_embedding = self.in_layer(x.float()) # self.in_layer 是一个投影层，例如 nn.Linear(3 * P, d_model)，用于将 patch 向量映射为 Transformer/GPT 接收的 token embedding。(B, N, d_model)
        
        # 类似 ViT (Vision Transformer) 中将图像切成 patch：这里是将时间序列“切片”为 patch；每个 patch 编码了 trend/seasonal/residual 的结构；然后送入 GPT/Transformer 主干模型，作为 token 序列进行学习；



            
        outs = self.prompt_pool(pre_prompted_embedding) # Semantic Space Informed Prompting
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

               

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B*M*3, -1))
            
            
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B,m=M,c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss
            

        

        
        
        outputs = outputs * stdev[:,:,:M]
        outputs = outputs + means[:,:,:M]

        return outputs,res





    










