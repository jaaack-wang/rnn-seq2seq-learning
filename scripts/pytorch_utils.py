'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: 
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from string import ascii_lowercase


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)
            

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def metrics(Y, Ypred):
    pw_cmp = (Y == Ypred).float()
    # batch-wise pair-wise overlap rate
    batch_overlap_rate = pw_cmp.mean(dim=0)
    
    # overlap_rate and abosulate accuracy
    overlap_rate = batch_overlap_rate.mean().item()
    abs_correct = (batch_overlap_rate == 1.0)
    abs_accu = abs_correct.float().mean().item()
    
    if pw_cmp.dim() <= 1:
        min_idx = pw_cmp.argmin(0)
        if pw_cmp[min_idx] == 0:
            pw_cmp[min_idx:] = 0
        
    else:
        for col_idx, min_idx in enumerate(pw_cmp.argmin(0)):
            if pw_cmp[min_idx, col_idx] == 0:
                pw_cmp[min_idx:, col_idx] = 0
                
    # consecutive overlap rate 
    cons_overlap_rate = pw_cmp.mean().item()
    
    return abs_accu, cons_overlap_rate, overlap_rate


def train_loop(model, dataloader, optimizer, 
               criterion, teacher_forcing_ratio):
    model.train()
    aggr_perf = {"loss": 0.0, 
                 "abosulate accuracy": 0.0, 
                 "consecutive overlap rate": 0.0, 
                 "overlap rate": 0.0}
    
    for X, Y in dataloader:
        seq_len, batch_size = Y.shape
        seq_len -= 1 # logits does not have <s> 
        
        X = X.to(model.device)
        Y = Y.to(model.device)
        optimizer.zero_grad()
        logits, _ = model(X, Y, teacher_forcing_ratio)
        
        Ypred = logits.view(seq_len, batch_size, -1).argmax(2)
        accu, covr, ovr = metrics(Y[1:-1], Ypred[:-1])
        loss = criterion(logits, Y[1:].view(-1))
        
        aggr_perf["loss"] += loss.item()
        aggr_perf["abosulate accuracy"] += accu
        aggr_perf["consecutive overlap rate"] += covr
        aggr_perf["overlap rate"] += ovr
        
        loss.backward(); optimizer.step()
    
    aggr_perf = {k:v/len(dataloader) for k,v in aggr_perf.items()}
    return aggr_perf
    

def evaluate(model, dataloader, criterion, 
             per_seq_len_performance=False):
    model.eval()
    
    if per_seq_len_performance:
        seq_len = set(X.shape[0] for X, _ in dataloader)
        assert len(seq_len) == len(dataloader), "Each batch" \
        " must contain sequences of a specific length. "
        
        perf_log = dict()
        
        
    aggr_perf = {"loss": 0.0, 
                 "abosulate accuracy": 0.0, 
                 "consecutive overlap rate": 0.0, 
                 "overlap rate": 0.0}
    
    with torch.no_grad():
        for X, Y in dataloader:
            seq_len, batch_size = Y.shape
            seq_len -= 1 # logits does not have <s> 
            
            X = X.to(model.device)
            Y = Y.to(model.device)
            logits, _ = model(X, Y, 
                              teacher_forcing_ratio=0.0)
            
            Ypred = logits.view(seq_len, batch_size, -1).argmax(2)
            accu, covr, ovr = metrics(Y[1:-1], Ypred[:-1])
            loss = criterion(logits, Y[1:].view(-1))
            
            aggr_perf["loss"] += loss.item()
            aggr_perf["abosulate accuracy"] += accu
            aggr_perf["consecutive overlap rate"] += covr
            aggr_perf["overlap rate"] += ovr
            
            if per_seq_len_performance:
                perf_log[f"Len-{seq_len-1}"] = {"loss": loss.item(), 
                                                "abosulate accuracy": accu, 
                                                "consecutive overlap rate": covr, 
                                                "overlap rate": ovr}
            
    aggr_perf = {k:v/len(dataloader) for k,v in aggr_perf.items()}
    
    if per_seq_len_performance:
        perf_log[f"Aggregated"] = aggr_perf
        
        return aggr_perf, perf_log
            
    return aggr_perf



def train_and_evaluate(model, train_dl, eval_dl, 
                       criterion, optimizer, 
                       saved_model_fp, acc_threshold=0.9, 
                       print_freq=5, max_epoch_num=100, 
                       train_acc_exit=0.9999, 
                       eval_acc_exit=0.995,
                       teacher_forcing_ratio=1.0):
    
    log = dict()
    best_acc = acc_threshold
    epoch, train_acc, eval_acc = 0, 0, 0
    while (epoch < max_epoch_num) and (eval_acc != 1.0) and (
        train_acc < train_acc_exit or eval_acc < eval_acc_exit):
        
        epoch += 1
        
        train_perf = train_loop(model, train_dl, optimizer, 
                                criterion, teacher_forcing_ratio)
        train_acc = train_perf['abosulate accuracy']
        
        eval_perf = evaluate(model, eval_dl, criterion)
        eval_acc = eval_perf['abosulate accuracy']
        
        if epoch % print_freq == 0:
            
            print(f"Current epoch: {epoch}, \ntraining performance: " \
                  f"{train_perf}\nevaluation performance: {eval_perf}\n")
            
        log[f"Epoch#{epoch}"] = {"Train": train_perf, "Eval": eval_perf}
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), saved_model_fp)
    
    if best_acc > acc_threshold:
        log["Best eval accu"] = best_acc
        print(saved_model_fp + " saved!\n")
        model.load_state_dict(torch.load(saved_model_fp))
        
    return log


def test_eval(data, batch_size, shuffle, out_dist=False):
    test_dl = create_dataloader(data, in_vocab, out_vocab)
