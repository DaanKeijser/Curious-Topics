import sys
seed = int(sys.argv[4])

import random
random.seed(a=seed)
# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
# setting torch seeds
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
# general imports
import json
import pickle
import time
import numpy as np
np.random.seed(seed)
# custom imports
import agent

###############################################################################
# evaluation function
###############################################################################

def evaluate(epoch, split='val'):
    listener.eval()
    speaker.eval()
    if split=='val':
        batchlist = val_batchlist
    elif split=='test':
        batchlist = test_batchlist
    n_batches = len(batchlist)
    start_time = time.time()
    li_eval_loss = np.empty(n_batches)
    li_eval_acc = np.empty(n_batches)
    sp_eval_loss = np.empty(n_batches)
    sp_eval_acc = np.empty(n_batches)
    batch_size = np.empty(n_batches)

    batch = 0

    while batch < n_batches:
        language_input, visual_input, targets = load_val_batch(dict_words_boxes,
                                                        batchlist[batch],
                                                        word_to_ix,
                                                        device)

        obj_guesses = listener(language_input, visual_input)
        obj_guess_values = obj_guesses.detach()

        word_guesses = speaker(visual_input, obj_guess_values)
        li_loss = criterion(obj_guesses, targets)
        li_eval_acc[batch], batch_size[batch] = calc_accuracy(obj_guesses, targets)
        li_eval_loss[batch] = li_loss.item() * batch_size[batch]
        li_eval_acc[batch] *= batch_size[batch] #avg weighted for differing batchsizes

        sp_loss = criterion(word_guesses, language_input)
        sp_eval_loss[batch] = sp_loss.item() * batch_size[batch]
        sp_eval_acc[batch],_ = calc_accuracy(word_guesses, language_input)
        sp_eval_acc[batch] *= batch_size[batch] #avg weighted for differing batchsizes

        batch += 1
        if batch % printerval == 0:
            print('| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | l.L {:6.4f} | l.A {:5.4f} | s.L {:6.4f} | s.A {:5.4f} |'.format(
                epoch, batch, n_batches, (time.time() - start_time),
                np.sum(li_eval_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(li_eval_acc[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_eval_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_eval_acc[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch])))

    avg_li_eval_loss = np.sum(li_eval_loss) / np.sum(batch_size)
    avg_li_eval_acc = np.sum(li_eval_acc) / np.sum(batch_size)
    avg_sp_eval_loss = np.sum(sp_eval_loss) / np.sum(batch_size)
    avg_sp_eval_acc = np.sum(sp_eval_acc) / np.sum(batch_size)

    if split == 'val':
        print('-' * 89)
        print("overall performance on validation set:")
        print('| L.loss {:8.4f} | L.acc. {:8.4f} |'.format(
            avg_li_eval_loss,
            avg_li_eval_acc))
        print('| S.loss {:8.4f} | S.acc. {:8.4f} |'.format(
            avg_sp_eval_loss,
            avg_sp_eval_acc))
        print('-' * 89)
    elif split == 'test':
        print('-' * 89)
        print("overall performance on test set:")
        print('| L.loss {:8.4f} | L.acc. {:8.4f} |'.format(
            avg_li_eval_loss,
            avg_li_eval_acc))
        print('| S.loss {:8.4f} | S.acc. {:8.4f} |'.format(
            avg_sp_eval_loss,
            avg_sp_eval_acc))
        print('-' * 89)
    return avg_li_eval_loss, avg_li_eval_acc, avg_sp_eval_loss, avg_sp_eval_acc

def load_val_batch(dict_words_boxes, batch, word_to_ix, device):
    # Loads the batches for the validation and test splits of the data
    language_input = []
    visual_input = []
    targets = []

    for img in batch:
        vggs = torch.load("/roaming/u865880/thesis_code/ha_bbox_vggs/"+img+".pt").to(device)
        for obj in dict_words_boxes[img]:
            language_input.append(get_word_ix(word_to_ix, dict_words_boxes[img][obj]["word"]))

            bbox_indices = []
            n = 0

            for obj_id in dict_words_boxes[img]:
                bbox_indices.append(ha_vggs_indices[img][obj_id][0])
                if obj_id == obj:
                    targets.append(n)
                n += 1
            visual_input.append(vggs[bbox_indices,:])

    lang_batch = torch.tensor(language_input, dtype=torch.long, device=device)
    vis_batch = torch.stack(visual_input)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return lang_batch, vis_batch, targets

def curiosity(i, o):
    # Calculates the curiosity values
    # i and o are torch tensors of shape [batch, ntokens]
    subjective_novelty = abs(i - o)
    plasticity = abs(o - o**2)
    return torch.mean((subjective_novelty * plasticity), dim=1)

def load_img(dict_words_boxes, ha_vggs_indices, img):
    vggs = torch.load("/roaming/u865880/thesis_code/ha_bbox_vggs/"+img+".pt").to(device) # Edit path
    # dict met obj ids als keys en een dictionary met words : '', bboxes :
    n = 0
    bbox_indices = []
    words = []
    for obj in dict_words_boxes[img]: # For every object in this image
        words.append(get_word_ix(word_to_ix, dict_words_boxes[img][obj]["word"]))
        bbox_indices.append(ha_vggs_indices[img][obj][0])
    visual_input = vggs[bbox_indices,:]
    language_input = torch.tensor(words, dtype=torch.long, device=device)
    return language_input, visual_input

def curious_look_at_img(dict_words_boxes, ha_vggs_indices, img):
    language_input, scene  = load_img(dict_words_boxes, ha_vggs_indices, img)
    # repeat scene n_objects times as input to listener
    visual_input = scene.expand(scene.size()[0], scene.size()[0], scene.size()[1])
    curiosity_targets = torch.eye(visual_input.size()[0], dtype=torch.float, device=device)
    # targets is simply 0, 1, ...., n because they are in order of appearance
    targets = torch.tensor([i for i in range(len(language_input))], dtype=torch.long, device=device)
    # word guesses by child - use as attention over word embeddings
    word_guesses = speaker(visual_input, curiosity_targets, apply_softmax=False)
    # only keep most likely words
    words = torch.argmax(word_guesses, dim=1)
    # give these as input to listener
    object_guesses = listener(words, visual_input)
    curiosity_values = curiosity(curiosity_targets, object_guesses)
    max_curious = torch.argmax(curiosity_values)
    return language_input[max_curious], scene, targets[max_curious]

def random_look_at_img(dict_words_boxes, ha_vggs_indices, img):
    language_input, scene = load_img(dict_words_boxes, ha_vggs_indices, img)
    # targets is simply 0, 1, ...., n because they are in order of appearance
    targets = torch.tensor([i for i in range(len(language_input))], dtype=torch.long, device=device)
    i = np.random.randint(len(targets))
    return language_input[i], scene, targets[i]

def load_select_obj(dict_words_boxes, ha_vggs_indices, img, setting):
    if setting == "random":
        return random_look_at_img(dict_words_boxes, ha_vggs_indices, img)
    elif setting == "curious":
        return curious_look_at_img(dict_words_boxes, ha_vggs_indices, img)

def calc_accuracy(guesses, targets, average=True):
    """
    in: log probabilities for C classes (i.e. candidate nrs), target 'class'
    indices (from 0 up-to-and-icluding C-1) (object position in your case)
    """
    score = 0
    guess = torch.argmax(guesses.data, 1)

    for i in range(targets.data.size()[0]):
        if guess.data[i] == targets.data[i]:
            score += 1

    if average:
        return score/targets.data.size()[0], targets.data.size()[0]
    else:
        return score, targets.data.size()[0]

###############################################################################
# Training function
###############################################################################

def train():
    listener.train()
    speaker.train()
    start_time = time.time()
    n_batches = len(batches)
    li_train_loss = np.empty(n_batches)
    li_train_accuracy = np.empty(n_batches)
    sp_train_loss = np.empty(n_batches)
    sp_train_accuracy = np.empty(n_batches)
    batch_size = np.empty(n_batches)

    batch = 0

    # batches shuffled during training
    while batch < n_batches:
        language_batch = [] # All word indices in the batch?
        visual_batch = [] # All vgg vectors in the batch?
        target_batch = [] # All target word indices in the batch?

        for img in batches[batch]:
            language_input, visual_input, target = load_select_obj(dict_words_boxes, ha_vggs_indices, img,
                setting)
            language_batch.append(language_input)
            visual_batch.append(visual_input)
            target_batch.append(target)
        language_input = torch.stack(language_batch)
        visual_input = torch.stack(visual_batch)
        targets = torch.stack(target_batch)

        speaker_optimizer.zero_grad()
        listener_optimizer.zero_grad()

        obj_guesses = listener(language_input, visual_input)

        # Saves the batch length for weighted mean accuracy:
        batch_size[batch] = len(batches[batch])

        loss = criterion(obj_guesses, targets)
        loss.backward() # backward pass
        listener_optimizer.step() # adapting the weights

        # Loss/accuracy times batch size for weighted average over epoch:
        li_train_loss[batch] = loss.item() * batch_size[batch]
        li_train_accuracy[batch],_ = calc_accuracy(obj_guesses, targets, average=False)

        obj_guess_values = obj_guesses.detach()

        word_guesses = speaker(visual_input, obj_guess_values)

        speaker_loss = criterion(word_guesses, language_input)
        speaker_loss.backward()
        speaker_optimizer.step()

        # Loss/accuracy times batch size for weighted average over epoch:
        sp_train_loss[batch] = speaker_loss.item() * batch_size[batch]
        sp_train_accuracy[batch],_ = calc_accuracy(word_guesses, language_input, average=False)

        batch += 1
        if batch % printerval == 0:
            print('| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | l.L {:6.4f} | l.A {:5.4f} | s.L {:6.4f} | s.A {:5.4f} |'.format(
                epoch, batch, n_batches, (time.time() - start_time),
                np.sum(li_train_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(li_train_accuracy[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_train_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_train_accuracy[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch])))

    avg_li_train_loss = np.sum(li_train_loss) / np.sum(batch_size)
    avg_li_train_acc = np.sum(li_train_accuracy) / np.sum(batch_size)
    avg_sp_train_loss = np.sum(sp_train_loss) / np.sum(batch_size)
    avg_sp_train_acc = np.sum(sp_train_accuracy) / np.sum(batch_size)

    print('-' * 89)
    print("overall performance on training set:")
    print('| L.loss {:8.4f} | L.acc. {:8.4f} |'.format(
        avg_li_train_loss,
        avg_li_train_acc))
    print('| S.loss {:8.4f} | S.acc. {:8.4f} |'.format(
        avg_sp_train_loss,
        avg_sp_train_acc))
    print('-' * 89)
    return avg_li_train_loss, avg_li_train_acc, avg_sp_train_loss, avg_sp_train_acc

#################################################################################
# Interpret command line arguments
#################################################################################

print("\nSys.argv:",sys.argv)
batchsize = int(sys.argv[1])
lr = float(sys.argv[2])
setting = sys.argv[3]

#################################################################################
# DATA PREPROCESSING
#################################################################################

def make_vocab(data_dict):
    '''
    Returns a vocabulary and a vocabulary with frequencies
    '''

    freq = {}
    for file in data_dict.keys():
        for obj in data_dict[file].keys():
            word = data_dict[file][obj]["word"]
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

    vocab = list(freq.keys())

    return vocab, freq

def make_ix_table(vocabulary):
    word_to_ix = {"<UNK>":0}
    index = 1

    for word in vocabulary:
        word_to_ix[word] = index
        index += 1

    return word_to_ix

def no_of_objs(data_dict, data_split):
    '''
    Returns a dictionary with the
    number of objects per image
    for a train/validation/test
    split of the data
    '''

    no_of_objs = {}

    for file in data_split:
        if len(data_dict[file]) not in no_of_objs:
            no_of_objs[len(data_dict[file])] = []
            no_of_objs[len(data_dict[file])].append(file)
        else:
            no_of_objs[len(data_dict[file])].append(file)

    return no_of_objs

def get_word_ix(word_to_ix, word):
    if word in word_to_ix:
        return word_to_ix[word]
    else:
        return word_to_ix["<UNK>"]

def dict_to_batches(no_objs_split, bsz):
    '''
    Returns a list of batches. A batch is a
    batch-size lists of file/img ids, of imgs
    containing the same amount of objects.
    The batches are shuffled so that batches
    of different amounts of objects follow
    each other.
    '''
    batch_list = []

    for num in no_objs_split.keys():
        batch_list.extend([no_objs_split[num][x:x+bsz] for x in range(0, len(no_objs_split[num]), bsz)])

    np.random.shuffle(batch_list,)

    return batch_list

###############################################################################
# ACTUAL CODE (NO MORE DEFINITIONS)
###############################################################################

# Object vgg indices (object information)
with open("/roaming/u865880/thesis_code/ha_vgg_indices.json", "rb") as input_file:
    ha_vggs_indices = json.load(input_file)

# Regular data (dictionary with all images and their object ids, corresponding words)
with open("/roaming/u865880/thesis_code/dict_words_boxes.json", "rb") as input_file:
    dict_words_boxes = json.load(input_file)

# Train split, image ids
with open("/roaming/u865880/thesis_code/train_data.txt", "rb") as fp:
    train_data = pickle.load(fp)

# Validation split, image ids
with open("/roaming/u865880/thesis_code/validation_data.txt", "rb") as fp:
    validation_data = pickle.load(fp)

# Test split, image ids
with open("/roaming/u865880/thesis_code/test_data.txt", "rb") as fp:
    test_data = pickle.load(fp)

###############################################################################
# PREPROCESSING
###############################################################################

# Makes a vocabulary of the entire set of objects:
vocab, freq = make_vocab(dict_words_boxes)

# Gives an index number to every word in the vocabulary:
word_to_ix = make_ix_table(vocab)

# Returns a dictionary with the number of objects per image:
no_objs = no_of_objs(dict_words_boxes, train_data)

# Returns a list of batch-size batches:
# A batch contains images with the same no. of objs
batches = dict_to_batches(no_objs, batchsize)

no_objs_val = no_of_objs(dict_words_boxes, validation_data)
val_batchlist = dict_to_batches(no_objs_val, batchsize)
#test set:
no_objs_test = no_of_objs(dict_words_boxes, test_data)
test_batchlist = dict_to_batches(no_objs_test, batchsize)

ntokens = len(word_to_ix.keys())
print("ntokens:",ntokens)

###############################################################################
# SPECIFY MODEL
###############################################################################
# these are the sizes Anna Rohrbach uses. she uses a batch size of 40.
#n_objects = 100
object_size = 4096 # Length vgg vector?
att_hidden_size = 256 # Number of hidden nodes
wordemb_size = 256 # Length word embedding
nonlin = "sigmoid"
print("hidden layer size:", att_hidden_size)

epochs = 40

device = torch.device('cuda') # Device = GPU

# Makes the listener part of the model:
listener = agent.Listener(object_size, ntokens, wordemb_size,
        att_hidden_size, nonlinearity=nonlin).to(device)

# Makes the speaker part of the model:
speaker = agent.Speaker(object_size, ntokens, att_hidden_size, nonlinearity=nonlin).to(device)

# Loss function: binary cross entropy
criterion = nn.CrossEntropyLoss(size_average = True)

###############################################################################
# TRAIN LOOP
###############################################################################
# Print after this many batches:
printerval = 100

print("parameters of listener agent:")
for param in listener.parameters():
    print(type(param.data), param.size())
listener_optimizer = optim.Adam(listener.parameters(), lr=lr)

print("parameters of speaker agent:")
for param in speaker.parameters():
    print(type(param.data), param.size())
speaker_optimizer = optim.Adam(speaker.parameters(), lr=lr)


# Creating numpy arrays to store loss and accuracy
# for train, validation, and test splits
listener_train_loss = np.empty(epochs)
listener_train_acc = np.empty(epochs)
speaker_train_loss = np.empty(epochs)
speaker_train_acc = np.empty(epochs)
listener_val_loss = np.empty(epochs)
listener_val_acc = np.empty(epochs)
speaker_val_loss = np.empty(epochs)
speaker_val_acc = np.empty(epochs)
listener_test_loss = np.empty(epochs)
listener_test_acc = np.empty(epochs)
speaker_test_loss = np.empty(epochs)
speaker_test_acc = np.empty(epochs)

# At any point you can hit Ctrl + C to break out of training early.

# first run evaluate to get random baseline
li_val_loss, li_val_acc, sp_val_loss, sp_val_acc = evaluate(0)

try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        li_train_loss, li_train_acc, sp_train_loss, sp_train_acc = train()
        listener_train_loss[epoch-1], listener_train_acc[epoch-1] = li_train_loss, li_train_acc
        speaker_train_loss[epoch-1], speaker_train_acc[epoch-1] = sp_train_loss, sp_train_acc

        li_val_loss, li_val_acc, sp_val_loss, sp_val_acc = evaluate(epoch)
        listener_val_loss[epoch-1], listener_val_acc[epoch-1] = li_val_loss, li_val_acc
        speaker_val_loss[epoch-1], speaker_val_acc[epoch-1] = sp_val_loss, sp_val_acc

        li_test_loss, li_test_acc, sp_test_loss, sp_test_acc = evaluate(epoch,'test')
        listener_test_loss[epoch-1], listener_test_acc[epoch-1] = li_test_loss, li_test_acc
        speaker_test_loss[epoch-1], speaker_test_acc[epoch-1] = sp_test_loss, sp_test_acc

# To enable to hit Ctrl + C and break out of training:
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Saving the loss and accuracy numpy arrays:
np.save('loss_acc/final/li_train_loss_{}_{}_{}'.format(
    str(lr), setting, seed), listener_train_loss)
np.save('loss_acc/final/li_train_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_train_acc)
np.save('loss_acc/final/sp_train_loss_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_train_loss)
np.save('loss_acc/final/sp_train_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_train_acc)
np.save('loss_acc/final/li_val_loss_{}_{}_{}'.format(
    str(lr), setting, seed), listener_val_loss)
np.save('loss_acc/final/li_val_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_val_acc)
np.save('loss_acc/final/sp_val_loss_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_val_loss)
np.save('loss_acc/final/sp_val_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_val_acc)
np.save('loss_acc/final/li_test_loss_{}_{}_{}'.format(
    str(lr), setting, seed), listener_test_loss)
np.save('loss_acc/final/li_test_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_test_acc)
np.save('loss_acc/final/sp_test_loss_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_test_loss)
np.save('loss_acc/final/sp_test_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_test_acc)
