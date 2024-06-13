import strtree


def test_init_strtree():
    tree = strtree.StringTree()


def test_build_tree():

    strings = ['Samsung X-500', 'Samsung SM-10', 'Samsung X-1100', 'Samsung F-10', 'Samsung X-2200',
               'AB Nokia 1', 'DG Nokia 2', 'THGF Nokia 3', 'SFSD Nokia 4', 'Nokia XG', 'Nokia YO']
    
    target = [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]

    tree = strtree.StringTree()
    tree.build(strings, target, min_precision=0.75, min_token_length=1, verbose=True)


def test_tree_methods():

    strings = ['Samsung X-500', 'Samsung SM-10', 'Samsung X-1100', 'Samsung F-10', 'Samsung X-2200',
               'AB Nokia 1', 'DG Nokia 2', 'THGF Nokia 3', 'SFSD Nokia 4', 'Nokia XG', 'Nokia YO']
    
    target = [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]

    tree = strtree.StringTree()
    tree.build(strings, target, min_precision=0.75, min_token_length=1, verbose=True)

    assert tree.filter(['OO Nokia 12']) == ['OO Nokia 12'] 
    assert tree.match(['OO Nokia 12']) == [1]
    assert tree.precision_score(['OO Nokia 12'], [1]) == 1.0
    assert tree.recall_score(['OO Nokia 12'], [1]) == 1.0