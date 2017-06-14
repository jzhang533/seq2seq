#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
inference the trained seq2seq model (attention model)
"""

import sys
import os
import theano
import numpy as np
import theano.tensor as T

np.set_printoptions(precision=8, suppress=True)
theano.config.floatX = 'float32'

ignore_unk = False
beam_size = 10

model_dir = "iter476000"
dict_dir = "data"

batch_size = 1
input_size = 620
hidden_size = 1000
alignment_hidden_size = hidden_size
maxout_size = hidden_size

#print("============load params from disk================")
encoder_emb = np.loadtxt("%s/encoder.emb" % model_dir)
decoder_emb = np.loadtxt("%s/decoder.emb" % model_dir)
encoder_rnn_w = np.loadtxt("%s/encoder_rnn.weights" % model_dir)
decoder_rnn_w = np.loadtxt("%s/decoder_rnn.weights.w" % model_dir)
decoder_rnn_u = np.loadtxt("%s/decoder_rnn.weights.u" % model_dir)
decoder_rnn_c = np.loadtxt("%s/decoder_rnn.weights.c" % model_dir)
decoder_rnn_att_w = np.loadtxt("%s/decoder_rnn.weights.att_w" % model_dir)
decoder_rnn_att_u = np.loadtxt("%s/decoder_rnn.weights.att_u" % model_dir)
decoder_rnn_att_v = np.loadtxt("%s/decoder_rnn.weights.att_v" % model_dir)

decoder_rnn_m_u = np.loadtxt("%s/decoder_rnn.weights.m_u" % model_dir)
decoder_rnn_m_v = np.loadtxt("%s/decoder_rnn.weights.m_v" % model_dir)
decoder_rnn_m_c = np.loadtxt("%s/decoder_rnn.weights.m_c" % model_dir)

fc_w = np.loadtxt("%s/fc.weights" % model_dir)
fc_b = np.loadtxt("%s/fc.bias" % model_dir)

def util_dot(inp, matrix):
    if 'int' in inp.dtype and inp.ndim==2:
        return matrix[inp.flatten()]
    elif 'int' in inp.dtype:
        return matrix[inp]
    elif 'float' in inp.dtype and inp.ndim == 3:
        shape0 = inp.shape[0]
        shape1 = inp.shape[1]
        shape2 = inp.shape[2]
        return T.dot(inp.reshape((shape0*shape1, shape2)), matrix)
    else:
        return T.dot(inp, matrix)

def softmax(x):
    if x.ndim == 2:
        e = T.exp(x)
        return e / T.sum(e, axis=1).dimshuffle(0, 'x')
    else:
        e = T.exp(x)
        return e/ T.sum(e)


ENC_V = theano.shared(encoder_emb.astype(theano.config.floatX))
DEC_V = theano.shared(decoder_emb.astype(theano.config.floatX))

new_fc_w = theano.shared(fc_w.astype(theano.config.floatX))
new_fc_b = theano.shared(fc_b.astype(theano.config.floatX))

# input shape: seq_len * batch
# an one batch example [[0], [1], [2]]
INPUT_SEQ = T.imatrix()

ENC_EMBS = ENC_V[INPUT_SEQ]

#use this to fake decoder rnn inputs
INPUT_TARGET_SEQ = T.imatrix()

#print("================load encoder normal rnn matrix==============")

offset = 0

Wr_np = encoder_rnn_w[offset:offset + input_size * hidden_size].reshape(
        hidden_size, input_size).transpose()
offset += input_size * hidden_size

Wi_np = encoder_rnn_w[offset: offset + input_size * hidden_size].reshape(
        hidden_size, input_size).transpose()
offset += input_size * hidden_size

Wh_np = encoder_rnn_w[offset: offset + input_size * hidden_size].reshape(
        hidden_size, input_size).transpose()
offset += input_size * hidden_size

Wr  = theano.shared(Wr_np.astype(theano.config.floatX))
Wi  = theano.shared(Wi_np.astype(theano.config.floatX))
Wh  = theano.shared(Wh_np.astype(theano.config.floatX))

Rr_np = encoder_rnn_w[offset:offset + hidden_size * hidden_size].reshape(
        hidden_size, hidden_size).transpose()
offset += hidden_size * hidden_size

Ri_np = encoder_rnn_w[offset: offset + hidden_size * hidden_size].reshape(
        hidden_size, hidden_size).transpose()
offset += hidden_size * hidden_size

Rh_np = encoder_rnn_w[offset: offset + hidden_size * hidden_size].reshape(
        hidden_size, hidden_size).transpose()
offset += hidden_size * hidden_size

Rr  = theano.shared(Rr_np.astype(theano.config.floatX))
Ri  = theano.shared(Ri_np.astype(theano.config.floatX))
Rh  = theano.shared(Rh_np.astype(theano.config.floatX))

#print("============load encoder reverse rnn matrix================")

R_Wr_np = encoder_rnn_w[offset:offset + input_size * hidden_size].reshape(
        hidden_size, input_size).transpose()
offset += input_size * hidden_size

R_Wi_np = encoder_rnn_w[offset: offset + input_size * hidden_size].reshape(
        hidden_size, input_size).transpose()
offset += input_size * hidden_size

R_Wh_np = encoder_rnn_w[offset: offset + input_size * hidden_size].reshape(
        hidden_size, input_size).transpose()
offset += input_size * hidden_size

R_Wr  = theano.shared(R_Wr_np.astype(theano.config.floatX))
R_Wi  = theano.shared(R_Wi_np.astype(theano.config.floatX))
R_Wh  = theano.shared(R_Wh_np.astype(theano.config.floatX))

R_Rr_np = encoder_rnn_w[offset:offset + hidden_size * hidden_size].reshape(
        hidden_size, hidden_size).transpose()
offset += hidden_size * hidden_size

R_Ri_np = encoder_rnn_w[offset: offset + hidden_size * hidden_size].reshape(
        hidden_size, hidden_size).transpose()
offset += hidden_size * hidden_size

R_Rh_np = encoder_rnn_w[offset: offset + hidden_size * hidden_size].reshape(
        hidden_size, hidden_size).transpose()
offset += hidden_size * hidden_size

R_Rr  = theano.shared(R_Rr_np.astype(theano.config.floatX))
R_Ri  = theano.shared(R_Ri_np.astype(theano.config.floatX))
R_Rh  = theano.shared(R_Rh_np.astype(theano.config.floatX))

#print("================load encoder normal rnn bias==============")

Bwr_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
Bwi_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
Bwh_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
Brr_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
Bri_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
Brh_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size

Bwr  = theano.shared(Bwr_np.astype(theano.config.floatX))
Bwi  = theano.shared(Bwi_np.astype(theano.config.floatX))
Bwh  = theano.shared(Bwh_np.astype(theano.config.floatX))
Brr  = theano.shared(Brr_np.astype(theano.config.floatX))
Bri  = theano.shared(Bri_np.astype(theano.config.floatX))
Brh  = theano.shared(Brh_np.astype(theano.config.floatX))

#print("================load encoder reverse rnn bias==============")

R_Bwr_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
R_Bwi_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
R_Bwh_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
R_Brr_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
R_Bri_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size
R_Brh_np = encoder_rnn_w[offset: offset + hidden_size]
offset += hidden_size

R_Bwr  = theano.shared(R_Bwr_np.astype(theano.config.floatX))
R_Bwi  = theano.shared(R_Bwi_np.astype(theano.config.floatX))
R_Bwh  = theano.shared(R_Bwh_np.astype(theano.config.floatX))
R_Brr  = theano.shared(R_Brr_np.astype(theano.config.floatX))
R_Bri  = theano.shared(R_Bri_np.astype(theano.config.floatX))
R_Brh  = theano.shared(R_Brh_np.astype(theano.config.floatX))

#print("================load attention decoder rnn==============")

De_W_np = decoder_rnn_w.reshape((input_size, 3, hidden_size))
De_R_np = decoder_rnn_u.reshape((hidden_size, 3, hidden_size))
De_C_np = decoder_rnn_c.reshape((2 * hidden_size, 3, hidden_size))
# reset
De_Wr_np = De_W_np[:, 0, :]
# update
De_Wi_np = De_W_np[:, 1, :]
# new memory
De_Wh_np = De_W_np[:, 2, :]

De_Wr  = theano.shared(De_Wr_np.astype(theano.config.floatX))
De_Wi  = theano.shared(De_Wi_np.astype(theano.config.floatX))
De_Wh  = theano.shared(De_Wh_np.astype(theano.config.floatX))

# reset
De_Rr_np = De_R_np[:, 0, :]
# update
De_Ri_np = De_R_np[:, 1, :]
# new memory
De_Rh_np = De_R_np[:, 2, :]

De_Rr  = theano.shared(De_Rr_np.astype(theano.config.floatX))
De_Ri  = theano.shared(De_Ri_np.astype(theano.config.floatX))
De_Rh  = theano.shared(De_Rh_np.astype(theano.config.floatX))

# reset
De_Cr_np = De_C_np[:, 0, :]
# update
De_Ci_np = De_C_np[:, 1, :]
# new memory
De_Ch_np = De_C_np[:, 2, :]

De_Cr  = theano.shared(De_Cr_np.astype(theano.config.floatX))
De_Ci  = theano.shared(De_Ci_np.astype(theano.config.floatX))
De_Ch  = theano.shared(De_Ch_np.astype(theano.config.floatX))

# linear for previous decoder hidden
At_W_np = decoder_rnn_att_w
# linear for encoder hidden
At_U_np = decoder_rnn_att_u
# project from alignment model size to a score
At_V_np = decoder_rnn_att_v 

At_W  = theano.shared(At_W_np.astype(theano.config.floatX))
At_U  = theano.shared(At_U_np.astype(theano.config.floatX))
At_V  = theano.shared(At_V_np.astype(theano.config.floatX))

De_M_U_np = decoder_rnn_m_u
De_M_V_np = decoder_rnn_m_v
De_M_C_np = decoder_rnn_m_c

De_M_U  = theano.shared(De_M_U_np.astype(theano.config.floatX))
De_M_V  = theano.shared(De_M_V_np.astype(theano.config.floatX))
De_M_C  = theano.shared(De_M_C_np.astype(theano.config.floatX))

#print("============load params end================")

x = ENC_EMBS
ones = theano.shared(np.ones((batch_size, hidden_size), dtype=theano.config.floatX))
h0 = theano.shared(np.zeros((batch_size, hidden_size), dtype=theano.config.floatX))

def recurrent(x_t, h_tm1):
    """
    hmmm.
    """
    i_t = T.nnet.sigmoid(T.dot(x_t, Wi) + T.dot(h_tm1, Ri) + Bwi + Bri)
    r_t = T.nnet.sigmoid(T.dot(x_t, Wr) + T.dot(h_tm1, Rr) + Bwr + Brr)

    h_prime_t = T.tanh(T.dot(x_t, Wh) + r_t * (T.dot(h_tm1, Rh) + Brh) + Bwh)
    h_t = (1.0 - i_t) * h_prime_t + i_t * h_tm1

    return [h_t]

h, updates = theano.scan(fn=recurrent, \
        sequences=x, outputs_info=h0, \
        n_steps=x.shape[0])

def reverse_recurrent(x_t, h_tm1):
    """
    hmmm.
    """
    i_t = T.nnet.sigmoid(T.dot(x_t, R_Wi) + T.dot(h_tm1, R_Ri) + R_Bwi + R_Bri)
    r_t = T.nnet.sigmoid(T.dot(x_t, R_Wr) + T.dot(h_tm1, R_Rr) + R_Bwr + R_Brr)

    h_prime_t = T.tanh(T.dot(x_t, R_Wh) + r_t * (T.dot(h_tm1, R_Rh) + R_Brh) + R_Bwh)
    h_t = (1.0 - i_t) * h_prime_t + i_t * h_tm1

    return [h_t]

r_h, updates = theano.scan(fn=reverse_recurrent, \
        sequences=x[::-1], outputs_info=h0, \
        n_steps=x.shape[0])

r_r_h = r_h[::-1]

source_seq_len = x.shape[0]

new_encoder_c = T.concatenate([h, r_r_h], axis=2).reshape((source_seq_len, 2 * hidden_size))

final_h = T.concatenate([h, r_r_h], axis=2).reshape((source_seq_len, batch_size, 2 * hidden_size))
#decoder_h0 = final_h[0, :, hidden_size : 2 * hidden_size].reshape((1, batch_size, hidden_size))
decoder_h0 = r_h[-1, :, :]

encoder_ff=theano.function(inputs=[INPUT_SEQ],outputs=[new_encoder_c], on_unused_input='ignore')
#print("final_h.ndim")
#print(final_h.ndim)
#print("decoder_h0.ndim")
#print(decoder_h0.ndim)
#print("At_W.ndim")
#print(At_W.ndim)

# reshaped final_h, will be used in computing dynamic context
reshaped_final_h = final_h.reshape((final_h.shape[0] * final_h.shape[1], final_h.shape[2]))
at_u_terms = T.dot(reshaped_final_h, At_U)

def attention_recurrent_generate(target_word_id, y_tm1, h_tm1):
    """
    hmmm.
    """
    x_t = DEC_V[y_tm1]
    x_t = x_t.reshape((x_t.shape[0], x_t.shape[2]))

    # computing dynamic context 
    at_w_terms = T.dot(h_tm1, At_W) 

    #print("at_w_terms.ndim")
    #print(at_w_terms.ndim)
    #print("h_tm1.ndim")
    #print(h_tm1.ndim)
    #print("At_W.ndim")
    #print(At_W.ndim)

    added = T.tile(at_w_terms, (256, 1))[0:at_u_terms.shape[0], :] + at_u_terms
    e_t = T.tanh(added)
    score_t = T.dot(e_t, At_V).reshape((final_h.shape[0], batch_size))

    # since nnet.softmax will do softmax row-wise, needs to transpose before and after
    a_t = T.nnet.softmax(score_t.transpose()).transpose()

    a_t_repeated = a_t.repeat(2 * hidden_size).reshape(
            (a_t.shape[0], a_t.shape[1], 2 * hidden_size))
    multiply_final_h = a_t_repeated * final_h
    context_t = T.sum(multiply_final_h, axis=0)

    r_t = T.nnet.sigmoid(T.dot(x_t, De_Wr) + T.dot(h_tm1, De_Rr) + T.dot(context_t, De_Cr))
    i_t = T.nnet.sigmoid(T.dot(x_t, De_Wi) + T.dot(h_tm1, De_Ri) + T.dot(context_t, De_Ci))

    h_prime_t = T.tanh(T.dot(x_t, De_Wh) + r_t * (T.dot(h_tm1, De_Rh)) + T.dot(context_t, De_Ch))
    h_t = (1.0 - i_t) * h_tm1 + i_t * h_prime_t

    m_u = T.dot(h_t, De_M_U)
    m_v = T.dot(x_t, De_M_V)
    m_c = T.dot(context_t, De_M_C)
    premaxout_t = m_u + m_v + m_c

    maxout_t = T.max(premaxout_t.reshape((premaxout_t.shape[0] * premaxout_t.shape[1] / 2, 2 )),
            axis=1).reshape((premaxout_t.shape[0], premaxout_t.shape[1] / 2))

    presoftmax_t = T.dot(maxout_t, fc_w) + fc_b
    y_t = T.cast(T.argmax(presoftmax_t).reshape((1, batch_size)), 'int32')

    return [y_t, h_t]

    #print("at_w_terms.ndim")
    #print(at_w_terms.ndim)
    #print("h_tm1.ndim")
    #print(h_tm1.ndim)
    #print("h_t.ndim")
    #print(h_t.ndim)
    #print("y_t.ndim")
    #print(y_t.ndim)
    #print("At_W.ndim")
    #print(At_W.ndim)
    
    #print("x_t.ndim")
    #print(x_t.ndim)
    #print("a_t.ndim")
    #print(a_t.ndim)
    #print("context_t.ndim")
    #print(context_t.ndim)


#    return [y_tm1, h_tm1]
#    return [y_t, h_t]

#""" 
[y, hiddens], updates = theano.scan(fn=attention_recurrent_generate, \
        sequences=INPUT_TARGET_SEQ, \
        n_steps=INPUT_TARGET_SEQ.shape[0], \
        outputs_info=[INPUT_TARGET_SEQ[:1], decoder_h0])


def step_prop(y_tm1, h_tm1, c):
    """
    hmmm.
    """
    x_t = DEC_V[y_tm1]
    #x_t = x_t.reshape((x_t.shape[0], x_t.shape[2]))

    # computing dynamic context 
    at_w_terms = T.dot(h_tm1, At_W) 

    #print("at_w_terms.ndim")
    #print(at_w_terms.ndim)
    #print("h_tm1.ndim")
    #print(h_tm1.ndim)
    #print("At_W.ndim")
    #print(At_W.ndim)

    if c.ndim == 2:
        c = c[:, None, :]
    
    source_len = c.shape[0]
    source_num = c.shape[1]
    target_num = h_tm1.shape[0]
    #reply cat 
    
    a = T.shape_padleft(at_w_terms)
    padding = [1] * at_w_terms.ndim
    b = T.alloc(np.float32(1), source_len, *padding)
    p_from_h = a * b

    p_from_c = util_dot(c,At_U).reshape((source_len, source_num, hidden_size))

    
    p =  p_from_h + p_from_c

    energy = T.exp(util_dot(T.tanh(p), At_V)).reshape((source_len, target_num))
    
    normalizer = energy.sum(axis=0)

    probs = energy / normalizer

    context_t = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)
    
    '''
    added = T.tile(at_w_terms, (256, 1))[0:at_u_terms.shape[0], :] + at_u_terms
    e_t = T.tanh(added)
    score_t = T.dot(e_t, At_V).reshape((final_h.shape[0], batch_size))

    # since nnet.softmax will do softmax row-wise, needs to transpose before and after
    a_t = T.nnet.softmax(score_t.transpose()).transpose()

    a_t_repeated = a_t.repeat(2 * hidden_size).reshape(
            (a_t.shape[0], a_t.shape[1], 2 * hidden_size))
    multiply_final_h = a_t_repeated * final_h
    context_t = T.sum(multiply_final_h, axis=0)
    '''

    r_t = T.nnet.sigmoid(T.dot(x_t, De_Wr) + T.dot(h_tm1, De_Rr) + T.dot(context_t, De_Cr))
    i_t = T.nnet.sigmoid(T.dot(x_t, De_Wi) + T.dot(h_tm1, De_Ri) + T.dot(context_t, De_Ci))

    h_prime_t = T.tanh(T.dot(x_t, De_Wh) + r_t * (T.dot(h_tm1, De_Rh)) + T.dot(context_t, De_Ch))
    h_t = (1.0 - i_t) * h_tm1 + i_t * h_prime_t

    m_u = T.dot(h_t, De_M_U)
    m_v = T.dot(x_t, De_M_V)
    m_c = T.dot(context_t, De_M_C)
    premaxout_t = m_u + m_v + m_c

    maxout_t = T.max(premaxout_t.reshape((premaxout_t.shape[0] * premaxout_t.shape[1] / 2, 2 )),
            axis=1).reshape((premaxout_t.shape[0], premaxout_t.shape[1] / 2))
    
    presoftmax_t = T.dot(maxout_t, new_fc_w) + new_fc_b

    #presoftmax_t = T.dot(h_t, new_fc_w) + new_fc_b
    #y_t = T.cast(T.argmax(presoftmax_t).reshape((1, batch_size)), 'int32')
    soft_max_res = softmax(presoftmax_t)

    return [soft_max_res, h_t]


last_state = T.matrix("last_state", dtype=theano.config.floatX)
encoder_c = T.matrix("encoder_c", dtype=theano.config.floatX)
last_word = T.lvector("last_word")

next_y, next_hidden = step_prop(last_word, last_state, encoder_c)
step_ff = theano.function(inputs=[last_word, last_state, encoder_c], outputs=[next_y, next_hidden],on_unused_input='ignore')


#""" 

#y, hiddens = attention_recurrent_generate(INPUT_TARGET_SEQ, INPUT_TARGET_SEQ[:1], decoder_h0)

y = T.flatten(y)

'''
ff = theano.function(inputs=[INPUT_SEQ, INPUT_TARGET_SEQ],
        outputs=[y, hiddens],
        on_unused_input='ignore')
'''

input_dict = {}
with open("%s/source.vocab" % dict_dir) as f:
    idx = 0
    for line in f.readlines():
        input_dict[line.strip()] = idx
        idx += 1

vocab = []
with open("%s/target.vocab" % dict_dir) as f:
    for line in f.readlines():
        vocab.append(line.strip())

EOS_ID = 2
UNK_ID = 3


#sentence = "当 我 走 在 这里 的 每 一 条 街道"
#sentence = sentence.decode("utf8").encode("gbk")

#sys.stdout.write("> ")
sys.stdout.flush()
sentence = sys.stdin.readline().strip()
#sentence = sentence.decode("utf8").encode("gbk")
while sentence:
    words = sentence.split(" ")
    ids = [input_dict.get(w, UNK_ID) for w in words]

    ids = list(ids)
#    print(" ".join([str(x) for x in ids]))

    input_ids = np.array(ids, dtype=np.int32)
    input_ids = input_ids.reshape((input_ids.shape[0], 1))

    # since go_id = 1, we will create an all go_id input_target_ids
    input_target_ids = np.ones((128, batch_size), dtype=np.int32)
    c1 = encoder_ff(input_ids)
    
    last_word_1 = input_target_ids[0]
    last_state_1 = c1[0][0, -hidden_size:].reshape((1,hidden_size))
    beam = beam_size

    costs = [0.0]
    base_costs = [0.0]
    trans = [[]]

    fin_trans = []
    fin_costs =[]
    

    for k in range(3*len(ids)):

        if beam == 0:
            break;
        
        last_word_1 = (np.array(map(lambda t : t[-1], trans))
                        if k > 0
                        else input_target_ids[0])
        #print "last_word_1"
        #print last_word_1
        #print last_word_1.shape
        
        #print "last_state"
        #print last_state_1.shape
        
        #print last_state_1
        next_y, next_hidden = step_ff(last_word_1, last_state_1,c1[0])

        #print (next_y.shape)
        #print next_hidden

        #last_state_1 = next_hidden
        
        log_prob = np.log(next_y)

        if ignore_unk:
            log_prob[:,UNK_ID] -= np.inf

        next_costs = np.array(costs)[:, None] - log_prob
        #next_costs = np.array(base_costs)[:, None] - log_prob

        flatten_next_costs = next_costs.flatten()
        
        best_costs_indices = np.argpartition( flatten_next_costs.flatten(), beam )[:beam]
        
        #print "best costt indices"
        #print  best_costs_indices
        #print(best_costs_indices.dtype)
        #print(best_costs_indices.shape)
        
        voc_size = log_prob.shape[1]
        #print (voc_size.dtype)
        trans_indices = best_costs_indices / voc_size
        word_indices = best_costs_indices % voc_size
        
        #print (trans_indices.dtype)
        #print (trans_indices.shape)
        #print "trans indices"
        #print (trans_indices)

        #print (word_indices.dtype)
        #print (word_indices.shape)
        #print "word indices"
        #print (word_indices)

        costs = flatten_next_costs[best_costs_indices]

        new_trans = [[]] * beam
        new_costs = np.zeros(beam)
        new_states = np.zeros((beam, hidden_size), dtype="float32")
        inputs = np.zeros(beam, dtype="int64")
        
        new_next_states = np.zeros((beam, hidden_size), dtype="float32")
        #print len(last_state_1)
        for i, (orig_idx, next_word, next_cost) in enumerate(
                zip(trans_indices, word_indices, costs)):
            #print orig_idx
            #print next_word
            #print "last: ", trans[orig_idx]
            #print "next", next_word
            #print "cost", next_cost
            new_trans[i] = trans[orig_idx] + [next_word]
            new_costs[i] = next_cost
            new_states[i] = last_state_1[orig_idx]
            new_next_states[i] = next_hidden[orig_idx]
            inputs[i] = next_word


        #next_y_2, new_states = step_ff(inputs, new_states)
    
        trans = []
        costs = []
        indices = []
        
        #print "new states shape"
        #print new_states.shape

        for i in range(beam):
            if new_trans[i][-1] != EOS_ID:
                trans.append(new_trans[i])
                costs.append(new_costs[i])
                indices.append(i)
            else:
                beam -= 1
                fin_trans.append(new_trans[i])
                fin_costs.append(new_costs[i])

        #print "trans"
        #print trans
        #print "beam"
        #print beam
        #print "indices"
        #print indices
        #laste_state_1 = map(lambda x : x[indices], new_states)
        last_state_1 = new_next_states[indices]
        #last_state_1 = new_states[indices]


    
    if( len(fin_trans) == 0 ):
        fin_trans = trans
        fin_costs = costs

    fin_trans = np.array(fin_trans)[np.argsort(fin_costs)]
    fin_costs = np.array(sorted(fin_costs))

    #for i in range(len(fin_trans)):
    for i in range(1):
        str_trans = " ".join([vocab[j] for j in fin_trans[i]][:-1])
        print (str_trans)
        #print (fin_costs[i])





    '''
    decoder_all_result = ff(input_ids, input_target_ids)

    result = decoder_all_result[0]
    decoder_hidden = decoder_all_result[1]

#    print(result)
#    print(decoder_hidden)

    result = list(result)
    if EOS_ID in result:
        result = result[:result.index(EOS_ID)]

    output = " ".join([vocab[i] for i in result])
#    output = output.decode("gbk").encode("utf8")
    print(output)
    print("> ", end = "")
    '''
    sys.stdout.flush()
    sentence = sys.stdin.readline().strip()
#    sentence = sentence.decode("utf8").encode("gbk")

