import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def calc_centroid(tensor):
    # Inputs Shape(N, 9 , 128, 128)
    # Return Shape(N, 9 ,2)
    input = tensor.float() + 1e-10
    n, l, h, w = input.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = input.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    center_x = input.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    output = torch.cat([center_y, center_x], 2)
    # output = torch.cat([center_x, center_y], 2)
    return output


class F1Score(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super(F1Score, self).__init__()
        #self.device = device
        self.name_list = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                        'left_ear',
                        'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace', 'neck',
                        'cloth']
        self.F1_name_list = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                        'left_ear',
                        'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace', 'neck',
                        'cloth']

        self.TP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.TN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.recall = {x: 0.0 + 1e-20
                       for x in self.F1_name_list}
        self.precision = {x: 0.0 + 1e-20
                          for x in self.F1_name_list}
        self.F1_list = {x: []
                        for x in self.F1_name_list}
        self.F1 = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}

        self.recall_overall_list = {x: []
                                    for x in self.F1_name_list}
        self.precision_overall_list = {x: []
                                       for x in self.F1_name_list}
        self.recall_overall = 0.0
        self.precision_overall = 0.0
        self.F1_overall = 0.0

    def forward(self, predict, labels):
      
        part_name_list = {6: 'eyebrow1', 7: 'eyebrow2', 4: 'eye1', 5: 'eye2',
                          6: 'nose', 11: 'u_lip', 10: 'i_mouth', 12: 'l_lip'}
        F1_name_list_parts = ['eyebrow1', 'eyebrow2',
                              'eye1', 'eye2',
                              'nose', 'u_lip', 'i_mouth', 'l_lip']
        TP = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        FP = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        TN = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        FN = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        pred = predict.argmax(dim=1, keepdim=False)
        # ground = labels.argmax(dim=1, keepdim=False)
        ground = labels.long()
        #print(ground.shape,labels.shape)
        assert ground.shape == pred.shape
        for i in range(1, 9):
            TP[part_name_list[i]] += ((pred == i) * (ground == i)).sum().tolist()
            TN[part_name_list[i]] += ((pred != i) * (ground != i)).sum().tolist()
            FP[part_name_list[i]] += ((pred == i) * (ground != i)).sum().tolist()
            FN[part_name_list[i]] += ((pred != i) * (ground == i)).sum().tolist()

        self.TP['mouth_all'] += (((pred == 6) + (pred == 7) + (pred == 8)) *
                                 ((ground == 6) + (ground == 7) + (ground == 8))
                                 ).sum().tolist()
        self.TN['mouth_all'] += (
                (1 - ((pred == 6) + (pred == 7) + (pred == 8)).float()) *
                (1 - ((ground == 6) + (ground == 7) + (ground == 8)).float())
        ).sum().tolist()
        self.FP['mouth_all'] += (((pred == 6) + (pred == 7) + (pred == 8)) *
                                 (1 - ((ground == 6) + (ground == 7) + (ground == 8)).float())
                                 ).sum().tolist()
        self.FN['mouth_all'] += ((1 - ((pred == 6) + (pred == 7) + (pred == 8)).float()) *
                                 ((ground == 6) + (ground == 7) + (ground == 8))
                                 ).sum().tolist()

        for r in ['eyebrow1', 'eyebrow2']:
            self.TP['eyebrows'] += TP[r]
            self.TN['eyebrows'] += TN[r]
            self.FP['eyebrows'] += FP[r]
            self.FN['eyebrows'] += FN[r]

        for r in ['eye1', 'eye2']:
            self.TP['eyes'] += TP[r]
            self.TN['eyes'] += TN[r]
            self.FP['eyes'] += FP[r]
            self.FN['eyes'] += FN[r]

        for r in ['u_lip', 'i_mouth', 'l_lip']:
            self.TP[r] += TP[r]
            self.TN[r] += TN[r]
            self.FP[r] += FP[r]
            self.FN[r] += FN[r]

        for r in ['nose']:
            self.TP[r] += TP[r]
            self.TN[r] += TN[r]
            self.FP[r] += FP[r]
            self.FN[r] += FN[r]

        for r in self.F1_name_list:
            self.recall[r] = self.TP[r] / (
                    self.TP[r] + self.FP[r])
            self.precision[r] = self.TP[r] / (
                    self.TP[r] + self.FN[r])
            self.recall_overall_list[r].append(self.recall[r])
            self.precision_overall_list[r].append(self.precision[r])
            self.F1_list[r].append((2 * self.precision[r] * self.recall[r]) /
                                   (self.precision[r] + self.recall[r]))
        return self.F1_list, self.recall_overall_list, self.precision_overall_list

    def output_f1_score(self):
        # print("All F1_scores:")
        for x in self.F1_name_list:
            self.recall_overall_list[x] = np.array(self.recall_overall_list[x]).mean()
            self.precision_overall_list[x] = np.array(self.precision_overall_list[x]).mean()
            self.F1[x] = np.array(self.F1_list[x]).mean()
            print("{}:{}\t".format(x, self.F1[x]))
        for x in self.F1_name_list:
            self.recall_overall += self.recall_overall_list[x]
            self.precision_overall += self.precision_overall_list[x]
        self.recall_overall /= len(self.F1_name_list)
        self.precision_overall /= len(self.F1_name_list)
        self.F1_overall = (2 * self.precision_overall * self.recall_overall) / \
                          (self.precision_overall + self.recall_overall)
        print("{}:{}\t".format("overall", self.F1_overall))
        return self.F1, self.F1_overall

    def get_f1_score(self):
        # print("All F1_scores:")
        for x in self.F1_name_list:
            self.recall_overall_list[x] = np.array(self.recall_overall_list[x]).mean()
            self.precision_overall_list[x] = np.array(self.precision_overall_list[x]).mean()
            self.F1[x] = np.array(self.F1_list[x]).mean()
        for x in self.F1_name_list:
            self.recall_overall += self.recall_overall_list[x]
            self.precision_overall += self.precision_overall_list[x]
        self.recall_overall /= len(self.F1_name_list)
        self.precision_overall /= len(self.F1_name_list)
        self.F1_overall = (2 * self.precision_overall * self.recall_overall) / \
                          (self.precision_overall + self.recall_overall)
        return self.F1, self.F1_overall


