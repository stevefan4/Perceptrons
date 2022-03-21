############################################################
# Imports
############################################################

import perceptrons_data as data

############################################################
# Section 1: Perceptrons
############################################################
def dot(i, j):
    if len(i) != len(j):
        return 0
    return sum(ele[0] * ele[1] for ele in zip(i, j))

class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        weights = {}
        for epoch in range(iterations): 
            for feat, y in examples:
                for key in weights:
                    if key not in feat:
                        feat[key] = 0
                for key, val in feat.items():
                    if key not in list(weights.keys()):
                        weights[key] = 0
                weights = dict((sorted(weights.items())))
                feat = dict(sorted(feat.items()))
                y_hat = dot(list(feat.values()), list(weights.values()))
                if y_hat <= 0:
                    y_hat = False
                else: 
                    y_hat = True
                if y_hat != y: 
                    for key in weights:
                        if key in feat:
                            if y == False: 
                                weights[key] = weights[key] - feat[key]
                            else: 
                                weights[key] = weights[key] + feat[key]
        self.w = weights

    def predict(self, x):
        for key in self.w:
            if key not in x:
                x[key] = 0
        y_hat = dot(list(x.values()), list(self.w.values()))
        if y_hat <= 0:
            return False
        else: 
            return True

class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        weights = {}
        for epoch in range(iterations):
            for feat, y in examples:
                if y not in list(weights.keys()):
                    weights[y] = {}   
                for ele in list(weights[list(weights.keys())[0]].keys()):
                    if ele not in feat:
                        feat[ele] = 0
                for key, val in feat.items(): 
                    for ele in weights.values(): 
                        if key not in list(ele.keys()):
                            ele[key] = 0            
                weights = dict((sorted(weights.items())))
                feat = dict(sorted(feat.items()))
                max_y = -float('inf')
                for label, weight in weights.items():
                    weight = dict(sorted(weight.items()))
                    y_hat = dot(list(feat.values()), list(weight.values()))
                    if y_hat > max_y: 
                        max_y = y_hat
                        max_label = label
                if max_label != y:
                    for key, value in sorted(list(weights[max_label].items())):
                        weights[max_label][key] = value - feat[key]
                    for key, value in sorted(list(weights[y].items())):
                        weights[y][key] = value + feat[key]
        self.w = weights
        
    def predict(self, x):
        max_y = -float('inf')
        x = dict(sorted(x.items()))
        for label, weights in list(self.w.items()):
            weights = dict(sorted(weights.items()))
            y_hat = dot(list(x.values()), list(weights.values()))
            if y_hat > max_y:
                max_y = y_hat
                max_label = label
        return max_label

############################################################
# Section 2: Applications
############################################################
class IrisClassifier(object):

    def __init__(self, data):
        train = []
        for features, y in data:
            value = {}
            value['x1'] = features[0]
            value['x2'] = features[1]
            value['x3'] = features[2]
            value['x3'] = features[3]
            train.append((value, y))
        p = MulticlassPerceptron(train, 20)
        self.p = p

    def classify(self, instance):
        value = {}
        value['x1'] = instance[0]
        value['x2'] = instance[1]
        value['x3'] = instance[2]
        value['x3'] = instance[3]       
        return self.p.predict(value)

class DigitClassifier(object):

    def __init__(self, data):
        train = []
        for features, y in data:
            feat = range(0, len(features))
            value = dict(zip(feat, features))
            train.append((value, y))
        p = MulticlassPerceptron(train, 4)
        self.p = p

    def classify(self, instance):
        feat = range(0, len(instance))
        value = dict(zip(feat, list(instance)))
        return self.p.predict(value)
            
class BiasClassifier(object):

    def __init__(self, data):
        train = []
        for feat, y in data: 
            value = {}
            value['x1'] = feat-1
            train.append((value, y))
        p = BinaryPerceptron(train, 10)
        self.p = p
                    
    def classify(self, instance):
        value = {}
        value['x1'] = instance-1
        return self.p.predict(value)

class MysteryClassifier1(object):

    def __init__(self, data):
        train = []
        for feat, y in data:
            value = {}
            value['x1'] = feat[0]**2 + feat[1]**2
            value['x2'] = 1
            train.append((value, y))
        p = BinaryPerceptron(train, 15)
        self.p = p

    def classify(self, instance):
        value = {}
        value['x1'] = instance[0]**2 + instance[1]**2
        value['x2'] = 1
        return self.p.predict(value)
    
class MysteryClassifier2(object):

    def __init__(self, data):
        train = []
        for feat, y in data:
            value = {}
            value['x1'] = feat[0] * feat[1] * feat[2]
            train.append((value, y))
        p = BinaryPerceptron(train, 20)
        self.p = p

    def classify(self, instance):
        value = {}
        value['x1'] = instance[0] * instance[1] * instance[2]
        return self.p.predict(value)
