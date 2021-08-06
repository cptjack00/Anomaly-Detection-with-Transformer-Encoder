import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn import metrics

from data_loader import CustomDataset
from models import make_autoencoder_model, make_transformer_model
from train import create_dataloader, create_mask
from utils import get_args, process_config, save_config

try:
    args = get_args()
    config = process_config(args.config)
except:
    print("Missing or invalid arguments")
    exit(0)
dataset = CustomDataset(config, train=False)
data_loader = create_dataloader(dataset, config)

autoencoder_model = make_autoencoder_model(
    seq_len=config["autoencoder_dims"], d_model=config["d_model"])
autoencoder_model.load_state_dict(torch.load(
    config["checkpoint_dir"] + config["best_auto_model"]))
autoencoder_model.float()
autoencoder_model.eval()
encoder = autoencoder_model.encoder

mask = create_mask(config)
trans_model = make_transformer_model(N=config["num_stacks"], d_model=config["d_model"], l_win=config["l_win"],
                                     d_ff=config["d_ff"], h=config["num_heads"], dropout=config["dropout"]).float()
trans_model.load_state_dict(torch.load(
    config["checkpoint_dir"] + config["best_trans_model"]))
trans_model.eval()

loss = torch.nn.MSELoss()
n_test = dataset.rolling_windows.shape[0]
recon_loss = np.zeros(n_test)
for i, batch in enumerate(data_loader):
    src = encoder(batch["input"].float())
    trg = encoder(batch["target"].float())
    out = trans_model(src, src_mask=mask)
    for j in range(config["batch_size"]):
        try:
            recon_loss[i * config["batch_size"] + j] = loss(
                out[j, config["pre_mask"]:config["post_mask"], :], trg[j, config["pre_mask"]:config["post_mask"], :])
        except:
            pass

idx_anomaly_test = dataset.data["idx_anomaly_test"]
anomaly_index = []
test_labels = np.zeros(n_test)
for i in range(len(idx_anomaly_test)):
    idx_start = idx_anomaly_test[i] - (config["l_win"] + 1)
    idx_end = idx_anomaly_test[i] + (config["l_win"] + 1)
    if idx_start < 0:
        idx_start = 0
    if idx_end > n_test:
        idx_end = n_test
    anomaly_index.append(np.arange(idx_start, idx_end))
    test_labels[idx_start:idx_end] = 1


def return_anomaly_idx_by_threshold(test_anomaly_metric, threshold):
    # test_list = np.squeeze(np.ndarray.flatten(test_anomaly_metric))
    idx_error = np.squeeze(np.argwhere(test_anomaly_metric > threshold))

    if len(idx_error.shape) == 0:
        idx_error = np.expand_dims(idx_error, 0)

    return list(idx_error)


def augment_detected_idx(idx_detected_anomaly, anomaly_index):
    n_anomaly = len(anomaly_index)
    idx_detected_anomaly_extended = list(idx_detected_anomaly)
    for i in range(n_anomaly):
        # print(idx_detected_anomaly)
        for j in idx_detected_anomaly:
            if j in anomaly_index[i]:
                in_original_detection = set(idx_detected_anomaly_extended)
                currect_anomaly_win = set(anomaly_index[i])
                idx_detected_anomaly_extended = idx_detected_anomaly_extended + \
                    list(currect_anomaly_win - in_original_detection)
                # print(j)
                break

    return list(np.sort(idx_detected_anomaly_extended))


def count_TP_FP_FN(idx_detected_anomaly, test_labels):
    n_TP = 0
    n_FP = 0
    #n_detection = len(idx_detected_anomaly)
    # for i in range(n_detection):
    for i in idx_detected_anomaly:
        # if test_labels[idx_detected_anomaly[i]] == 1:
        if test_labels[i] == 1:
            n_TP = n_TP + 1
        else:
            n_FP = n_FP + 1

    idx_undetected = list(set(np.arange(len(test_labels))
                              ) - set(idx_detected_anomaly))
    n_FN = 0
    for i in idx_undetected:
        if test_labels[i] == 1:
            n_FN = n_FN + 1
    n_TN = len(test_labels) - n_TP - n_FP - n_FN
    return n_TP, n_FP, n_FN, n_TN


def compute_precision_and_recall(idx_detected_anomaly, test_labels):
    # compute true positive
    n_TP, n_FP, n_FN, n_TN = count_TP_FP_FN(idx_detected_anomaly, test_labels)

    if n_TP + n_FP == 0:
        precision = 1
    else:
        precision = n_TP / (n_TP + n_FP)
    recall = n_TP / (n_TP + n_FN)
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall)/(precision + recall)
    fpr = n_FP/(n_FP + n_TN)

    return precision, recall, F1, fpr, n_TP, n_FP, n_FN


def KQp(data, q):
    # UNTITLED3 Summary of this function goes here
    #   Detailed explanation goes here
    data2 = np.sort(data)  # sap xep tang dan
    n = np.shape(data2)[0]  # kich thuoc
    p = 1-q  # q tu xet, dat bang smth 0.05 0.025 0.01
    h = math.sqrt((p*q)/(n+1))
    KQ = 0
    for i in range(1, n+1):
        a = ((i/n)-p)/h
        b = (((i-1)/n)-p)/h
        TP = (norm.cdf(a)-norm.cdf(b))*data2[i-1]  # normcdf thu trong matlab
        KQ = KQ+TP
    #KQp = KQ;
    return KQ


# for i in [0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
#     print("KQp is:", KQp(recon_loss, i), "with q =", i)

n_threshold = 20
precision = np.zeros(n_threshold)
recall = np.zeros(n_threshold)
F1 = np.zeros(n_threshold)
precision_aug = np.zeros(n_threshold)
recall_aug = np.zeros(n_threshold)
F1_aug = np.zeros(n_threshold)
fpr_aug = np.zeros(n_threshold)
i = 0
threshold_list = np.linspace(np.amin(recon_loss), np.amax(
    recon_loss), n_threshold, endpoint=False)
print(threshold_list)
threshold_list = np.flip(threshold_list)
print(threshold_list)
for threshold in threshold_list:
    # print(threshold_list[i])
    print(i)
    idx_detection = return_anomaly_idx_by_threshold(recon_loss, threshold)
    # augment the detection using the ground truth labels
    # a method to discount the factor one anomaly appears in multiple consecutive windows
    # introduced in "Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications"
    idx_detection_augmented = augment_detected_idx(
        idx_detection, anomaly_index)
    precision_aug[i], recall_aug[i], F1_aug[i], fpr_aug[i], _, _, _ = compute_precision_and_recall(idx_detection_augmented,
                                                                                                   test_labels)
    i = i + 1
    #print(precision, recall, F1)


print("\nAugmented detection:")
print("Best F1 score is {}".format(np.amax(F1_aug)))
idx_best_threshold = np.squeeze(np.argwhere(F1_aug == np.amax(F1_aug)))
print("Best threshold is {}".format(threshold_list[idx_best_threshold]))
best_thres = np.min(threshold_list[idx_best_threshold])
print("At this threshold, precision is {}, recall is {}".format(precision_aug[idx_best_threshold],
                                                                recall_aug[idx_best_threshold]))

# Now select a threshold
threshold = best_thres
q_list = [0.99, 0.9, 0.1]
temp = math.inf
q_best = 0
closest_thres = 0
for q in q_list:
    temp_thres = KQp(recon_loss, q)
    #print(temp_thres,abs(temp_thres - threshold))
    if abs(temp_thres - threshold) < temp:
        temp = abs(temp_thres - threshold)
        q_best = q
        KQp_thres = temp_thres

start = time.time()
print("Closest KQp threshold is {} at q = {}".format(KQp_thres, q_best))
config["q_best"] = q_best
idx_detection = return_anomaly_idx_by_threshold(recon_loss, KQp_thres)
# print(idx_detection)
idx_detection_augmented = augment_detected_idx(idx_detection, anomaly_index)
# print(anomaly_index_lstm)
# print(idx_detection_augmented)
precision, recall, F1, _, n_TP, n_FP, n_FN = compute_precision_and_recall(idx_detection_augmented,
                                                                          test_labels)
# print("\nPR evaluation using KQE:")
# print("Precision: {}".format(precision))
config["precision"] = precision
# print("Recall: {}".format(recall))
config["recall"] = recall
# print("F1: {}".format(F1))
config["F1"] = F1
# print("TP: {}".format(n_TP))
# print("FP: {}".format(n_FP))
# print("FN: {}".format(n_FN))

tpr = np.insert(recall_aug, [0, n_threshold], [0, 1])
fpr = np.insert(fpr_aug, [0, n_threshold], [0, 1])
auc = metrics.auc(fpr, tpr)
print("AUC =", auc)
config["AUC"] = auc
lw = 2
plt.plot(fpr, tpr, color="darkorange", lw=lw,
         label="ROC curve (area = %0.4f)" % auc)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Augmented Receiver operating characteristic of " +
          config["auto_dataset"])
plt.legend(loc="lower right")
plt.savefig(config["result_dir"] + "augmentedroc.pdf")
config["inference_time"] = (time.time() - start) / 60
save_config(config)
