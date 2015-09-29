import theano
import theano.tensor as T

def rank_loss(scores):
    # Images
    diag   = T.diag(scores)
    diff_img = scores - diag.dimshuffle(0, 'x') + 1
    max_img = T.maximum(0, diff_img)
    triu_img = T.triu(max_img, 1)
    til_img  = T.tril(max_img, -1)
    res_img = T.sum(triu_img) + T.sum(til_img)

    # Sentences
    diff_sent = scores.T - diag.dimshuffle(0, 'x') + 1
    max_sent = T.maximum(0, diff_sent)
    triu_sent = T.triu(max_sent, 1)
    til_sent  = T.tril(max_sent, -1)
    res_sent = T.sum(triu_sent) + T.sum(til_sent)
    
    return T.log(T.sum(scores) + 0.01)

def rank_loss_old(scores):
    '''
    Example algorithm of loss function:
    scores   = np.array([[1,2], [3,4]])
    scores_T = np.transpose(scores)
    cost = 0
    d_img_idxs = range(2)
    for k in d_img_idxs:
        rank_imgs = rank_sents = 0
        for l in [x for x in d_img_idxs if x != k]:
            S_kl, S_kk = scores[k,l], scores[k,k]
            rank_imgs += max(0, S_kl - S_kk + 1)
        for l in [x for x in d_img_idxs if x != k]:
            S_lk, S_kk = scores_T[k,l], scores_T[k,k]
            rank_sents += max(0, S_lk - S_kk + 1)
        cost += rank_imgs + rank_sents
    print cost
    '''

    
    def loss_loop(k, num_imgs, scores):
        inner_loop_idxs = T.neq(T.arange(scores.shape[0]), k).nonzero()[0]
        inner_loop_idxs.name = 'inner_loop_idxs'
    
        results_imgs, _ = theano.scan(
            fn=lambda l, scores: T.maximum(0, scores[k,l] - scores[k,k] + 1),
            sequences=[inner_loop_idxs],
            non_sequences=[scores]
        )
        results_sents, _ = theano.scan(
            fn=lambda l, scores: T.maximum(0, scores[k,l] - scores[k,k] + 1),
            sequences=[inner_loop_idxs],
            non_sequences=[scores.T]
        )
        results_imgs = T.sum(results_imgs)
        results_sents = T.sum(results_sents)
        
        return results_imgs + results_sents

    num_imgs = scores.shape[0]
    num_imgs.name = 'num_imgs'
    loss_loop_idxs = T.arange(scores.shape[0])
    loss_loop_idxs.name = 'loss_loop_idxs'

    results, _ = theano.scan(
        fn=loss_loop,
        outputs_info=None,
        sequences=[loss_loop_idxs],
        non_sequences=[num_imgs, scores]
    )
    return T.log(T.sum(results))

    '''
    
    '''