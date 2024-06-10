import re

import numpy as np


class Pattern:
    """Class for representing a pattern (a regular expression)."""
    def __init__(self, pattern):
        """Initialize a Pattern.

        Parameters
        ----------
        pattern : str | Pattern
            Str or Pattern object representing a regular expressions. It must be compatible with re.compile method.
        """
        if isinstance(pattern, str):
            self.str = pattern
            self.regex = re.compile(pattern)
        elif isinstance(pattern, Pattern):
            self.str = pattern.str
            self.regex = pattern.compile
        else:
            raise ValueError(f'Unknown type {type(pattern)} for Pattern __init__ method')
            
    def __str__(self):
        return self.str
    
    def __repr__(self):
        return f'Pattern("{self.str}")'

    def __copy__(self):
        return Pattern(self.str)

    def scores(self, strings, labels):
        """Calculate classification quality scores for pattern's matches and the labels.

        Pattern's matches are considered as "predictions" when calculating quality metrics.
        
        Parameters
        ----------
        strings : list[str]
            A list of strings.
        labels : list[int]
            A list of strings labels consisting of 0 and 1.

        Returns
        -------
        response : dict
            Contains keys: 'n_strings', 'total_positive', 'n_matches', 'precision', 'recall' and 'accuracy'.
        """
        regex = self.regex
        true_positive = 0
        true_negative = 0
        total_positive = sum(labels)
        n_strings = len(strings)
        n_matches = 0
        for i in range(n_strings):
            match = regex.search(strings[i]) is not None
            if match:
                n_matches += 1
                if labels[i] == 1:
                    true_positive += 1
            else:
                if labels[i] == 0:
                    true_negative += 1
        if n_matches == 0:
            precision = 0.
            accuracy = 0.
        else:
            precision = true_positive / n_matches
            accuracy = (true_positive + true_negative) / n_matches
        if total_positive == 0:
            recall = 0.
        else:
            recall = true_positive / total_positive
        response = {
            'n_strings': n_strings,
            'total_positive': total_positive,
            'n_matches': n_matches,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        return response

    def match(self, string):
        """Verify if the pattern matches the string."""
        return self.regex.search(string) is not None

    def filter(self, strings, labels=None):
        """Return strings and labels which match the pattern and which don't.

        Parameters
        ----------
        strings : list[str]
            Strings to filter.
        labels : list[int], default None
            Labels of strings.

        Returns
        -------
        matched_strings
            Strings which match the pattern.
        labels_of_matched
            Labels of strings which match the pattern. If no labels provided, an empty list.
        not_matched_strings
            Strings which don't match the pattern.
        labels_of_not_matched
            Labels of strings which don't match the pattern. If no labels provided, an empty list.
        """
        matched_strings = []
        labels_of_matched = []
        not_matched_strings = []
        labels_of_not_matched = []
        
        regex = self.regex
        for string_i in range(len(strings)):
            match = regex.search(strings[string_i]) is not None
            if match:
                matched_strings.append(strings[string_i])
                if labels is not None:
                    labels_of_matched.append(labels[string_i])
            else:
                not_matched_strings.append(strings[string_i])
                if labels is not None:
                    labels_of_not_matched.append(labels[string_i])
    
        return matched_strings, labels_of_matched, not_matched_strings, labels_of_not_matched


class PatternNode:
    """Class representing a node in a StringTree.

    Attributes
    ----------
    right : PatternNode
        Node with matching strings.
    left : PatternNode
        Node with non-matching strings.
    pattern : Pattern
        Attributed Pattern object.
    matches
        Strings which match a Pattern object attributed to the PatternNode.
    strings
        All strings attributed to the PatternNode.
    labels
        Labels of all strings attributed to the PatternNode. 
    scores
        Scores of an attributed pattern applied to the attributed strings.
    """
    def __init__(self, pattern: 'Pattern'):
        """Initialize a PatternNode.

        Parameters
        ----------
        pattern : Pattern
            A Pattern object attributed to a node.
        """
        self.right = None
        self.left = None
        self._pattern = pattern
        self._matches = None
        self._strings = None
        self._labels = None
        self._scores = None
    
    def __str__(self):
        return str(self.pattern)
    
    def __repr__(self):
        right_node = 'None' if self.right is None else f'PatternNode({self.right})'
        left_node = 'None' if self.left is None else f'PatternNode({self.left})'
        repr_ = [
            'PatternNode(',
            f'"{self._pattern}", ',
            f'right={right_node}, ',
            f'left={left_node}, ',
            f'n_strings={len(self._strings)}, ',
            f'precision={self._scores["precision"]}, '
            f'recall={self._scores["recall"]}'
            ')'
        ]
        repr_ = ''.join(repr_)
        return repr_

    def get_strings(self):
        """Get PatternNode.strings attribute"""
        if self._strings is not None:
            return self._strings.copy()
        return None

    def set_strings(self, strings):
        """Set PatternNode.strings attribute"""
        if self._strings is not None:
            raise ValueError("PatternNode.strings attribute is immutable once set")
        self._strings = strings.copy()

    def get_labels(self):
        """Get PatternNode.labels attribute"""
        if self._labels is not None:
            return self._labels.copy()
        return None

    def set_labels(self, labels):
        """Set PatternNode.labels attribute"""
        if self._labels is not None:
            raise ValueError("PatternNode.labels attribute is immutable once set")
        self._labels = labels.copy()

    def get_matches(self):
        """Get PatternNode.matches attribute"""
        if self._matches is not None:
            return self._matches.copy()
        return None

    def set_matches(self, matches):
        """Set PatternNode.matches attribute"""
        if self._matches is not None:
            raise ValueError("PatternNode.matches attribute is immutable once set")
        self._matches = matches.copy()

    def get_pattern(self):
        """Get PatternNode.pattern attribute"""
        if self._pattern is not None:
            return self._pattern  # Not a copy is returned!
        return None

    def set_pattern(self, pattern):
        """Set PatternNode.pattern attribute"""
        if self._pattern is not None:
            raise ValueError("PatternNode.pattern attribute is immutable once set")
        self._pattern = pattern.copy()

    def get_scores(self):
        """Get PatternNode.scores attribute"""
        if self._scores is not None:
            return self._scores.copy()
        return None

    def set_scores(self, scores):
        """Set PatternNode.scores attribute"""
        if self._scores is not None:
            raise ValueError("PatternNode.scores attribute is immutable once set")
        self._scores = scores.copy()

    pattern = property(get_pattern, get_pattern)
    strings = property(get_strings, set_strings)
    labels = property(get_labels, set_labels)
    matches = property(get_matches, set_matches)
    scores = property(get_scores, set_scores)


class StringTree:
    """A class for binary classification of strings with regular expressions. 
    
    Each node is an instance of the PatternNode class. It contains a regular expression and metadata. 

    Attributes
    ----------
    root : PatternNode
        The root PatternNode.
    leaves : list[PatternNode]
        List of all nodes.
    """
    def __init__(self):
        """Initialize a StringTree object."""
        self.root = None
        self._leaves = None

    def __repr__(self):
        return f'StringTree(root={self.root.__repr__()})'

    @staticmethod
    def _generate_tokens(strings, labels, length):
        """Generate tokens of given length from given strings.

        Tokens are parts of strings of a given length. 
        Tokens with "placeholders" like ".+", ".*", etc are also generated.
        Tokens are not generated for strings with label = 0.

        Parameters
        ----------
        strings : list[str]
            List of strings.
        labels : list[int]
            List of labels (0 or 1).
        length : int
            Length of tokens.

        Returns
        -------
        tokens : dict
            Dict containing tokens as keys and number of occurence among strings as values.
        """
        n = length
        ngrams = {}
        for string_i in range(len(strings)):
            if labels[string_i] == 0:  # Don't add ngramgs for negative target.
                continue
            string = strings[string_i]
            for i in range(len(string) - n + 1):
                token = string[i: i + n]
                if i == 0:
                    token = '^' + token
                elif i > 0:
                    token = '.+' + token
                if i < len(string) - n:
                    token = token + '.+'
                elif i == len(string) - n:
                    token = token + '$'
                if token not in ngrams:
                    ngrams[token] = {"count": 0}
                ngrams[token]["count"] += 1
    
                # You may add other tokens here,
                # f.e. with tokens ".*", "\d", etc
        return ngrams  

    @staticmethod
    def _combine_patterns(pattern1: 'Pattern', pattern2: 'Pattern'):
        """Return all possible combinations of pattern1 and pattern2.

        The method also takes into consideration placeholders in patterns, like ".+", ".*", etc. 
        For example, patterns like ".+A.+" and ".+B.+" are combined in the following way: [".+A.+B.+", ".+AB.+", ".+B.+A.+", ".+BA.+"]

        Returns
        -------
        patterns : list[Pattern]
            List of combined patterns.
        """
        patterns = []
        if pattern1.str == '':
            patterns.append(pattern2)
            return patterns
        if pattern2.str == '':
            patterns.append(pattern1)
            return patterns
        
        for (left_pattern, right_pattern) in [(pattern1, pattern2), (pattern2, pattern1)]:
            if left_pattern.str[-1] != '$' and right_pattern.str[0] != '^': 
                if left_pattern.str[-2:] in ['.+', '.*']:
                    patterns.append(Pattern(left_pattern.str[:-2] + right_pattern.str))
                    if right_pattern.str[:2] in ['.+', '.*']:
                        patterns.append(Pattern(left_pattern.str[:-2] + right_pattern.str[2:]))
                else:
                    patterns.append(Pattern(left_pattern.str + right_pattern.str))
                    if right_pattern.str[:2] in ['.+', '.*']:
                        patterns.append(Pattern(left_pattern.str + right_pattern.str[2:]))
                
        return patterns

    @staticmethod
    def _augment_pattern(strings, labels, current_pattern: 'Pattern', tokens):
        """Find best regular expression.

        Check each token added to the current_pattern for the given strings and labels and find the best new pattern.

        Parameters
        ----------
        strings
            A list of strings.
        labels
            A list of labels (0 and 1).
        current_pattern : Pattern
            Current pattern.
        tokens : list[str]
            List of tokens which should be combined with the current pattern.

        Returns
        -------
        pattern : Pattern
            Augmented pattern. If no such pattern was found, current_pattern is returned.
        """
        candidates_scores = []
        pattern_candidates = []
        for token in tokens:
            token_patterns = StringTree._combine_patterns(current_pattern, Pattern(token))
            for pattern in token_patterns:
                scores = pattern.scores(strings, labels)
                precision, recall, n_match = scores['precision'], scores['recall'], scores['n_matches']
                if precision + recall == 0:
                    f1_score = 0
                else:
                    f1_score =  2*precision*recall / (precision+recall) 
                candidates_scores.append(f1_score)
                pattern_candidates.append(pattern)
        if len(pattern_candidates) > 0:
            best_pattern_id = np.argmax(np.array(candidates_scores))
            best_pattern = pattern_candidates[best_pattern_id]
            return best_pattern
        return current_pattern

    def build(
            self, strings, labels, min_precision=0.5, min_token_length=1, 
            max_patterns=None, min_matches_leaf=1, min_strings_leaf=1, 
            verbose=False):
        """Build a StringTree.
        
        For the StringTree object being used, create nodes and corresponding patterns. Use provided strings and labels.

        Parameters
        ----------
        strings : list[str]
            List of strings.
        labels : list[int]
            List of labels (0 or 1).
        min_precision : float, default 0.5
            The minimal precision of a pattern in the tree.
        min_token_length : int, default 1
            The initial length of the pattern.
        max_patterns : int, default None
            The highest amount of patterns. Once the method finds more, it stops.
        min_matches_leaf : int, default 1
            The minimal amount of matches in one node.
        min_strings_leaf : int, default 1
            The minimal amount of strings in one node.
        verbose : bool, default False
            If to provide additinal text output.
        """
        if min_token_length <= 0:
            raise ValueError('min_token_length must not be <= 0')
        if min_precision < 0 or min_precision > 1:
            raise ValueError('min_precision must not be < 0 or > 1')
        if max_patterns is None:
            max_patterns = np.inf
            
        cur_strings = strings.copy()
        cur_labels = labels.copy()

        if verbose:
            print(f'Total: {len(strings)} strings with {sum(cur_labels)} positive labels')
        
        evaluation_queue = [(cur_strings, cur_labels)]
        leaves = []

        prev_node = None

        for (cur_strings, cur_labels) in evaluation_queue:
            if verbose:
                print(f'\nStart processing another {len(cur_strings)} of strings with {sum(cur_labels)} positive labels.')
            
            cur_pattern = Pattern('')
            scores = cur_pattern.scores(cur_strings, cur_labels)

            cur_node = PatternNode(cur_pattern)
            cur_node.scores = scores
            cur_node.strings = cur_strings
            cur_node.labels = cur_labels
            cur_node.matches = cur_strings
            
            precision = scores['precision']
            recall = scores['recall']
            n_matches = scores['n_matches']
            if verbose:
                print(f'Current pattern="{cur_pattern}". N matches: {n_matches}, Precision={precision}, Recall={recall}')

            first_run = True

            local_cur_strings, local_cur_labels = cur_strings, cur_labels
            while (precision < min_precision
                  and sum(local_cur_labels) > 0 
                  and n_matches > min_matches_leaf
                  and len(local_cur_strings) > min_strings_leaf):
                if first_run:
                    token_size = min_token_length
                else:
                    token_size = 1
                ngrams = StringTree._generate_tokens(local_cur_strings, local_cur_labels, token_size)
                tokens = list(ngrams.keys())

                best_pattern = StringTree._augment_pattern(local_cur_strings, local_cur_labels, cur_pattern, tokens)
                cur_pattern = best_pattern
                scores = cur_pattern.scores(local_cur_strings, local_cur_labels)

                precision = scores['precision']
                recall = scores['recall']
                n_matches = scores['n_matches']
                if verbose:
                    print(f'Current pattern="{cur_pattern}". N matches: {n_matches}, Precision={precision}, Recall={recall}')
                
                local_cur_strings, local_cur_labels, _, _ = \
                cur_pattern.filter(local_cur_strings, local_cur_labels)

                first_run = False

            cur_node = PatternNode(cur_pattern)
            cur_node.scores = scores
            cur_node.strings = cur_strings
            cur_node.labels = cur_labels

            cur_strings, cur_labels, not_matched_strings, labels_of_not_matched = \
            cur_pattern.filter(cur_strings, cur_labels)

            cur_node.matches = cur_strings
                
            if sum(labels_of_not_matched) > 0:
                evaluation_queue.append((not_matched_strings, labels_of_not_matched))
                
            if sum(cur_labels) > 0:
                leaves.append(cur_node)
                if verbose:
                    print('Last pattern was saved')
            
            if prev_node is None:
                self.root = cur_node
            else:
                prev_node.left = cur_node
            prev_node = cur_node

            if len(leaves) > max_patterns:
                if verbose:
                    print(f'Max patterns number of {max_patterns} is reached')
                break

        if verbose:
            print('\nFinished')

        self._leaves = leaves

    def get_leaves(self):
        """Get leaves attribute."""
        if self._leaves is not None:
            return self._leaves.copy()
        return None

    def set_leaves(self, leaves):
        """Set leaves attribute."""
        if self._leaves is not None:
            raise ValueError("StringTree.leaves attribute is immutable once set")
        self._leaves = leaves.copy()

    leaves = property(get_leaves, set_leaves)

    def filter(self, strings, return_nodes=False):
        """Return strings matching the tree and corresponding nodes.

        A string matches a tree if it matches at least one node.

        Parameters
        ----------
        strings : list[str]
            List of strings.
        return_nodes : bool, default False
            Flag indicating if to return nodes corresponding to the matched strings. 
            If False, only matched strings are returned.

        Returns
        -------
        matches : list[int]
            List containing matching strings.
        matched_nodes : list[PatternNode]
            List consisting of PatternNodes of matching strings. Returned only if return_nodes is True.
        """
        if self._leaves is None:
            raise ValueError("The StringTree was not built. Run StringTree.build method first.")

        matched_strings = []
        matched_nodes = []
        for string in strings:
            node = self.root
            match = node.pattern.match(string)
            while not match and node.left is not None:
                node = node.left
                match = node.pattern.match(string)
            if match:
                matched_strings.append(string)
                matched_nodes.append(node)

        if return_nodes:
            return matched_strings, matched_nodes
        return matched_strings

    def match(self, strings, return_nodes=False):
        """Return flags indicating if strings match the tree.

        A string matches a tree if it matches at least one node.

        Parameters
        ----------
        strings : list[str]
            List of strings.
        return_nodes : bool, default False
            Flag indicating if to return nodes corresponding to the matched strings. 
            If False, only matched strings are returned.

        Returns
        -------
        matches : list[int]
            List containing 1 (match) and 0 (no match) for each string.
        matched_nodes : list[PatternNode]
            List consisting of PatternNodes of matching strings. Returned only if return_nodes is True.
        """
        if self._leaves is None:
            raise ValueError("The StringTree was not built. Run StringTree.build method first.")

        matches = []
        matched_nodes = []
        for string in strings:
            node = self.root
            match = node.pattern.match(string)
            while not match and node.left is not None:
                node = node.left
                match = node.pattern.match(string)
            if match:
                matches.append(1)
                matched_nodes.append(node)
            else:
                matches.append(0)
                matched_nodes.append(None)

        if return_nodes:
            return matches, matched_nodes
        return matches

    def precision_score(self, strings, labels):
        """Calculate a precision score for given strings and labels."""
        matches = self.match(strings)
        pred_as_true = np.array(labels)[(np.array(matches) == 1)]
        
        if len(pred_as_true) > 0:
            precision_score = sum(pred_as_true) / len(pred_as_true)
        else:
            precision_score = 0
        return precision_score

    def recall_score(self, strings, labels):
        """Calculate a recall score for given strings and labels."""
        matches = self.match(strings)
        pred_as_true = np.array(labels)[(np.array(matches) == 1)]
        
        if len(pred_as_true) > 0:
            recall_score = sum(pred_as_true) / sum(labels)
        else:
            recall_score = 0
        return recall_score
