from baselines.markov0 import Markov0


def test_markov0_frequency_and_online_update():
    m = Markov0().fit([["a", "a", "b"], ["a", "c"]])  # a:3 b:1 c:1, total 5
    assert round(m.predict([])["a"], 3) == 0.6
    assert m.predict_top([], k=1)[0][0] == "a"          # most frequent
    m.update([], "b")                                    # online: b now 2, total 6
    assert round(m.predict([])["b"], 3) == round(2 / 6, 3)


def test_markov0_empty():
    assert Markov0().predict([]) == {} and Markov0().predict_top([]) == []
