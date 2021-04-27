import argparse
import math
import os

import numpy as np


class Evaluator:
    def __init__(self, gt_dict, res_dict, crange):
        self.evaluate_result = {}
        self.crange = crange
        # sort by key
        self._gt_dict = dict(
            sorted(gt_dict.items(), key=lambda d: d[0]))
        self._res_dict = dict(
            sorted(res_dict.items(), key=lambda d: d[0]))
        self._gt_score = np.array(list(self._gt_dict.values()))
        self._res_score = np.array(list(self._res_dict.values()))

    def show_result(self):
        for item in self.evaluate_result.keys():
            self._show_single_item(item)

    def _show_single_item(self, item):
        if item in self.evaluate_result:
            print("{} : {}".format(
                item.upper(),
                round(self.evaluate_result[item], 3)
            ))
        else:
            print("{} : NULL".format(item.upper()))
        # pass

    def evaluate(self):
        self.evaluate_result['mae'] = self._evaluate_MAE()
        self.evaluate_result['maod'] = self._evaluate_MAOD()
        self.evaluate_result['pearson_corr'] = self._evaluate_pearson_corr()
        self.evaluate_result['ndcg'] = self._evaluate_ndcg()
        self.evaluate_result['spearman_rank_corr'] = \
            self._evaluate_spearman_rank_corr()
        self.evaluate_result['kendall_rank_corr'] = \
            self._evaluate_kendall_rank_corr()
        self.evaluate_result['acc_crange{}'.format(self.crange)] = \
            self._evaluate_crange_acc()
        return self.evaluate_result

    def _evaluate_crange_acc(self):
        return np.sum((np.abs(self._gt_score-self._res_score) <
                       self.crange).astype(int))/len(self._gt_score)

    def _evaluate_MAE(self):
        return np.mean(np.abs(self._gt_score - self._res_score))

    def _evaluate_MAOD(self):
        sorted_gt = list(dict(
            sorted(self._gt_dict.items(), key=lambda d: d[1], reverse=True)))
        sorted_res = list(dict(
            sorted(self._res_dict.items(), key=lambda d: d[1], reverse=True)))
        gt_rank_dict = self._get_rank_dict(sorted_gt, self._gt_dict)
        res_rank_dict = self._get_rank_dict(sorted_res, self._res_dict)
        corr = 0
        n = max(len(sorted_gt), 1)
        for key in sorted_gt:
            corr += abs(gt_rank_dict[key] - res_rank_dict[key])
        corr = corr / (n * n)
        return corr

    def _evaluate_pearson_corr(self):
        res_avg = np.array([np.mean(self._res_score)] * len(self._res_dict))
        gt_avg = np.array([np.mean(self._gt_score)] * len(self._res_dict))
        corr = np.sum((self._res_score - res_avg)
                      * (self._gt_score - gt_avg)) \
            / np.sqrt(np.sum(np.power((self._res_score - res_avg), 2))) \
            / np.sqrt(np.sum(np.power((self._gt_score - gt_avg), 2)))
        return corr

    def _evaluate_ndcg(self):
        # sort by value
        sorted_gt_dict = dict(
            sorted(self._gt_dict.items(), key=lambda d: d[1], reverse=True))
        sorted_res_dict = dict(
            sorted(self._res_dict.items(), key=lambda d: d[1], reverse=True))
        sorted_gt = list(sorted_gt_dict.keys())
        sorted_res = list(sorted_res_dict.keys())
        gt_rank_dict = self._get_rank_dict(sorted_gt, self._gt_dict)
        res_rank_dict = self._get_rank_dict(sorted_res, self._res_dict)
        dcg = self._get_dcg(sorted_gt, res_rank_dict)
        idcg = self._get_dcg(sorted_gt, gt_rank_dict)
        ndcg = dcg / idcg
        return ndcg

    def _get_rank_dict(self, sorted_list, score_dict):
        rank_dict = {}
        n = float(len(self._res_dict))
        index = 0
        while(index < n):
            pre = index
            while (index + 1 < n and score_dict[sorted_list[index + 1]]
                   == score_dict[sorted_list[index]]):
                index += 1
            for i in range(pre, index + 1):
                rank_dict[sorted_list[i]] = n - (pre + index) / 2.0
            index += 1
        return rank_dict

    def _get_dcg(self, sorted_list, rank_dict):
        n = len(self._res_dict)
        dcg = 0
        for i in range(0, n):
            order = rank_dict[sorted_list[i]]
            dcg += order / math.log2(i + 2)
        return dcg

    def _evaluate_spearman_rank_corr(self):
        # sort by value
        sorted_gt = list(dict(
            sorted(self._gt_dict.items(), key=lambda d: d[1], reverse=True)))
        sorted_res = list(dict(
            sorted(self._res_dict.items(), key=lambda d: d[1], reverse=True)))
        gt_rank_dict = self._get_rank_dict(sorted_gt, self._gt_dict)
        res_rank_dict = self._get_rank_dict(sorted_res, self._res_dict)
        n = len(sorted_gt)
        corr = 0
        for key in gt_rank_dict:
            corr += math.pow((gt_rank_dict[key] - res_rank_dict[key]), 2)
        corr = 1 - 6 * corr / (n * (n * n - 1))
        return corr

    def _evaluate_kendall_rank_corr(self):
        # sort by value
        sorted_gt = list(dict(
            sorted(self._gt_dict.items(), key=lambda d: d[1])).keys())
        n = len(self._res_dict)
        nc = 0
        nd = 0
        for i in range(0, n):
            key_i = sorted_gt[i]
            for j in range(i + 1, n):
                key_j = sorted_gt[j]
                if (self._res_dict[key_i] == self._res_dict[key_j]
                        or self._gt_dict[key_i] == self._gt_dict[key_j]):
                    continue
                if (self._res_dict[key_i] < self._res_dict[key_j]
                        and self._gt_dict[key_i] < self._gt_dict[key_j]):
                    nc += 1
                else:
                    nd += 1
        if nc == nd:
            return 0
        sorted_res_score = list(dict(
            sorted(self._res_dict.items(), key=lambda d: d[1])).values())
        sorted_gt_score = list(dict(
            sorted(self._gt_dict.items(), key=lambda d: d[1])).values())
        n0 = n * (n - 1) / 2
        n1 = self._get_tied_value(sorted_gt_score)
        n2 = self._get_tied_value(sorted_res_score)
        return (nc - nd) / math.sqrt((n0 - n1) * (n0 - n2))

    def _get_tied_value(self, sorted_score):
        n = len(sorted_score)
        index = 0
        res = 0
        while(index < n):
            pre = index - 1
            while(index + 1 < n and (sorted_score[index + 1]
                                     == sorted_score[index])):
                index += 1
            if index - pre > 0:
                res += (index - pre) * (index - pre - 1) / 2
            index += 1
        return res


class EvaluatorInterval(Evaluator):
    def __init__(self, gt_interval_dict, res_dict):
        self._gt_interval_dict = gt_interval_dict
        gt_dict = {}
        for key in self._gt_interval_dict:
            assert len(self._gt_interval_dict[key]) == 2
            gt_dict[key] = (self._gt_interval_dict[key][0]
                            + self._gt_interval_dict[key][1]) / 2
        Evaluator.__init__(self, gt_dict, res_dict)

    def _hit_rate(self):
        cnt = 0
        for key in self._gt_interval_dict:
            if (self._res_dict[key] >= self._gt_interval_dict[key][0]
                    and self._res_dict[key] <= self._gt_interval_dict[key][1]):
                cnt += 1
        return cnt / len(self._res_dict)

    def _min_dis_sum(self):
        sum = 0
        for key in self._gt_interval_dict:
            if self._res_dict[key] > self._gt_interval_dict[key][1]:
                sum += self._res_dict[key] - self._gt_interval_dict[key][1]
            if self._res_dict[key] < self._gt_interval_dict[key][0]:
                sum += self._gt_interval_dict[key][0] - self._res_dict[key]
        return sum

    def evaluate_interval(self):
        res = self.evaluate()
        res['hit_rate'] = self._hit_rate()
        res['min_dis_sum'] = self._min_dis_sum()
        return res


class EvaluatorEX(object):
    def __init__(self):
        self.history = []

    @staticmethod
    def eval(y_pred, y_true, crange):
        ids = np.arange(0, len(y_true), dtype=int)
        ids = list(map(str, ids))
        y_pred_dict = dict(list(zip(ids, y_pred)))
        y_true_dict = dict(list(zip(ids, y_true)))
        evaluator = Evaluator(y_true_dict, y_pred_dict, crange)
        res = evaluator.evaluate()
        return res

    def add_record(self, res):
        self.history.append(res)

    def get_current_mean(self,):
        ans = dict([(k, []) for k in self.history[0]])
        n = len(self.history)
        for metrics in self.history:
            for k, v in metrics.items():
                ans[k].append(v)
        for k in ans:
            ans[k]=np.mean(ans[k]),np.std(ans[k])
        return ans


def main():
    parser = argparse.ArgumentParser(
        description="...")
    parser.add_argument(
        'gt_path',
        help="The ground truth path")
    parser.add_argument(
        'res_path',
        help="The result score path")
    parser.add_argument(
        'gt_range',
        help="The range of ground truth score.")
    parser.add_argument(
        'res_range',
        help="The range of result score.")
    args = parser.parse_args()
    res_dict = {}
    gt_dict = {}
    assert int(args.res_range) > 0 and int(args.gt_range) > 0
    with open(args.res_path, 'r') as f:
        for res in f.readlines():
            res = res.strip('\n')
            res_dict[res.split('\t')[0]] = float(res.split('\t')[1]) \
                / float(args.res_range) * 100
    with open(args.gt_path, 'r') as f:
        for gt in f.readlines():
            gt = gt.strip('\n')
            # if gt.split('\t')[0] in res_dict.keys():
            gt_dict[gt.split('\t')[0]] = \
                float(gt.split('\t')[1]) / float(args.gt_range) * 100
    assert res_dict.keys() == gt_dict.keys()
    eval = Evaluator(gt_dict, res_dict)
    eval_dict = eval.evaluate()
    print(eval_dict)
    eval.show_result()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
