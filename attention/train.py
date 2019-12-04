import os, sys
import torch.nn.functional as F
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from ut import *
#from dataset import TrainDataset
from sample import sampler
import pandas as pd
import numpy as np

global f2, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

# Model parameters
emb_dim = 300  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 64
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
f2 = 0.  # BLEU-4 score right now
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = 'checkpoint_attention.pth.tar' # path to checkpoint, None if none

def f2score(scores, targets, l,k):
    
    batch_size = scores.size(0)
    
    def get_score(target, y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(target, y_pred, beta=2, average='samples')
            
    y_pred = np.zeros((batch_size, 1103))
    for i in range(batch_size):
        for cls in scores[i,:l[i]]:
            if cls ==1104:
               break
            if cls <=1102:
                y_pred[i,cls] = 1
    #y_pred = np.concatenate(y_pred)
    
    target = np.zeros((batch_size, 1103))
    for i in range(batch_size):
        for cls in targets[i,:]:
            if cls ==1104:
                break
            if cls <=1102:
                target[i,cls] = 1
    #target = np.concatenate(target)
     
    f2 = get_score(target, y_pred)
    return f2
    



def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    top5f2 = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
       
#        print("caps_sorted",caps_sorted.size())
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        caps_sorted = caps_sorted.view(caps.size()[0], -1).to(device)
#        print("caps_sorted2",caps_sorted.size())
        targets = caps_sorted[:, 1:]
        targets = targets.type(torch.LongTensor).to(device)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
#        print("score1",scores.size())
#        print("targets1",targets.size())
        S = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        T = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        scores = S.data
        targets = T.data
        #scores = scores.to(device)
        #targets = targets.to(device)
        
        # Calculate loss
        
        
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
            
        roughscore = pad_packed_sequence(S, batch_first = True)
        roughscore = roughscore[0].argmax(dim=2)
        #print(roughscore)
        #print("rough",roughscore[0].argmax(dim=2))
        rought = pad_packed_sequence(T, batch_first = True)[0]
        #print(rought)
        #print("rough",rought[0])
        

        # Keep track of metrics
        top5 = accuracy(scores, targets,5)
        top5f = f2score(roughscore, rought,decode_lengths, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        top5f2.update(top5f, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'f2 {top5f2.val:.4f} ({top5f2.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                           batch_time=batch_time,
                                                                          top5f2=top5f2,
                                                                           loss=losses,
                                                                          top5=top5accs))



def caption_image_beam_search(encoder, decoder, image, beam_size = 3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = 1106

    encoder_out = image # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_out = encoder_out.unsqueeze(0)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[1103]]*k).to(device)  # (k, 1)
    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != 1104]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas
    
    


def validate(val_loader, encoder, decoder):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()
        
    vocab_size = 1106

    with torch.no_grad():
        # Batches
        f2 = []
        for i, (imgs, caps, caplens) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            #caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
                
            batch_size = imgs.size(0)
            for image_index in range(batch_size):    
                seq,_  = caption_image_beam_search(encoder, decoder, imgs[image_index], 4)
                seq = np.array(seq)
                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = torch.zeros(1103)
                y_pred = torch.zeros(1103)
                
           
                for cls1 in caps[image_index,:]:
                    if cls1<=1102:
                       targets[cls1] = 1
                    if cls1 == 1104:
                        break
            
                for cls2 in seq:
                    if cls2<=1102:
                       y_pred[cls2] = 1
                    if cls2 == 1104:
                        break
                
                def getscore(targets,y_pred):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UndefinedMetricWarning)
                        return fbeta_score(
                            targets, y_pred, beta=2)
                            
                f2.append(getscore(targets,y_pred))

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'f2score {f2:.4f})\t'
                      .format(i, len(val_loader),f2 = np.mean(f2)))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            

    return np.mean(f2)


"""
Training and validation.
"""



# Read word map
#    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
#    with open(word_map_file, 'r') as j:
#        word_map = json.load(j)
word_map = pd.read_csv('~/data/labels.csv')
mark = pd.DataFrame([[1103,'<start>'],[1104,'<end>'],[1105,'<pad>']])
mark.columns = word_map.columns
word_map.append(mark)
# Initialize / load checkpoint
if checkpoint is None:
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size= 1106,
                                   dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None

else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    f2 = checkpoint['f2']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    if fine_tune_encoder is True and encoder_optimizer is None:
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

# Move to GPU, if available
decoder = decoder.to(device)
encoder = encoder.to(device)

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_loader,  val_loader = sampler(0, 64)
# Epochs
for epoch in range(start_epoch, epochs):

    # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
    if epochs_since_improvement == 20:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        adjust_learning_rate(decoder_optimizer, 0.8)
        if fine_tune_encoder:
            adjust_learning_rate(encoder_optimizer, 0.8)
    #epoch = 0
    # One epoch's training
    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch)

    # One epoch's validation
    recent_f2 = validate(val_loader=val_loader,
                            encoder=encoder,
                            decoder=decoder)

    # Check if there was an improvement
    is_best = recent_f2 > f2
    f2 = max(f2, recent_f2)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        epochs_since_improvement = 0
  
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'f2': f2,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + 'attention' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)
