import theano
import theano.tensor as T

def rank_loss(X, scores):
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

    return T.sum(results) + 0 * T.sum(T.sum(X[0]))