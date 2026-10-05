"""Consensus invariants independent of any trained network or test labels."""
import unittest
import numpy as np
from tools.denice_peer_voting import vote,peer_orders


class PeerVotingTests(unittest.TestCase):
    def test_weighted_votes_and_self_ties(self):
        pred=np.array([[2,7,2],[7,2,7],[7,2,7]])
        labels=[2,7]
        np.testing.assert_array_equal(vote(pred,[1,1,1],labels,pred[0]),[7,2,7])
        np.testing.assert_array_equal(vote(pred,[2,1,1],labels,pred[0]),pred[0])
        np.testing.assert_array_equal(vote(pred,[0,0,0],labels,pred[0]),pred[0])
        # Per-sample router confidence can change a vote, while a common scale
        # on all alpha weights must not change the prediction.
        weight=np.array([[.8,.1,.8],[.1,.8,.1],[.1,.8,.1]])
        np.testing.assert_array_equal(vote(pred,weight,labels,pred[0]),[2,2,2])
        np.testing.assert_array_equal(vote(pred,weight*123,labels,pred[0]),[2,2,2])

    def test_sample_order_invariance(self):
        rng=np.random.default_rng(4);pred=rng.choice([2,7,11],size=(5,50))
        weights=rng.uniform(size=(5,50));order=rng.permutation(50)
        result=vote(pred,weights,[2,7,11],pred[0])
        np.testing.assert_array_equal(vote(pred[:,order],weights[:,order],[2,7,11],pred[0,order]),result[order])
        for col in range(50):
            self.assertEqual(vote(pred[:,col:col+1],weights[:,col:col+1],[2,7,11],pred[0,col:col+1])[0],result[col])

    def test_deterministic_graph_and_random_selection(self):
        peers=[9,4,2,1,6];alpha={1:.2,2:.3,4:.3,6:.1,9:.4}
        orders=peer_orders(1,peers,alpha)
        self.assertEqual(orders['alpha'],[9,2,4,6])
        self.assertEqual(orders,peer_orders(1,list(reversed(peers)),alpha))
        for order in orders.values():
            self.assertEqual(set(order),{9,2,4,6})
            self.assertTrue(set(order[:2]).issubset(order[:4]))

    def test_invalid_weights_rejected(self):
        for weight in ([1,-1],[1,float('nan')]):
            with self.assertRaises(ValueError):
                vote([[2],[7]],weight,[2,7],[2])


if __name__=='__main__':unittest.main()
