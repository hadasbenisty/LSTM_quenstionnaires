import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mat73
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scipy.io as spio
import sys
import classification_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from model import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

def balanceClusters(traininds, outcomevar):

    a, b = np.histogram(outcomevar[traininds], np.unique(outcomevar[traininds]))
    maxsz = max(a)

    traininds1=[0]
    for valbin in b:
        list = [i for i, val in enumerate(outcomevar[traininds] == valbin) if val]
        list = np.random.choice(list, size=maxsz, replace=True, p=None)
        traininds1 = np.concatenate((traininds1, list), axis=0)
    traininds1=traininds1[1:-1]
    traininds1 = traininds[traininds1]
    return traininds1
def norm_minmax(x):
    if len(np.unique(x)) == 1:
        return x
    inputx = x
    x[np.isnan(x)]=0
    m = np.min(x)
    M = np.max(x)
    if M-m != 0:
        x = (inputx - m) / (M - m)

    else:
        x = inputx
    x[np.isnan(inputx)] = 0
    return x

print("Gene expression")

# parsing input
lam = float(sys.argv[1])
Ngenes = int(sys.argv[2])
num_epoch=10000
FOLDS_NUM = 10
parent_dir = os.getcwd()


respath = "../../metabric_classificationPam50"
if not os.path.isdir(respath):
    os.mkdir(respath)
print("Directory '%s'" % respath)

print("Ngenes " + str(Ngenes))
print("lam " + str(lam))

datapath = '../../data/gene_expression/'


raw_meta_feat = mat73.loadmat(datapath+'Complete_METABRIC_Clinical_Features_Data.mat')
internal_param = raw_meta_feat["age_at_diagnosis"].astype(float)
internal_param = 2 * (norm_minmax(internal_param.reshape(-1, 1)) - 0.5)
outcomevar = raw_meta_feat["NOT_IN_OSLOVAL_Pam50Subtype"].astype(int).reshape(-1)
raw_gene_data = mat73.loadmat(datapath + 'metabric_exp' + '.mat')
raw_gene_inds = mat73.loadmat(datapath + 'gene_inds.mat')
raw_il = mat73.loadmat(datapath + 'illumina_id_to_gene_name.mat')

selected_genes = raw_gene_inds["isvalid"] & raw_gene_inds["ispam50"]
selected_genes = selected_genes[:raw_gene_data['data_mat'].shape[0]]
i = 0
Ngeneso=Ngenes
while Ngeneso > 0:
    if raw_gene_inds["isvalid"][i] and not (selected_genes[i]):
       selected_genes[i] = True
       Ngeneso -= 1
    i += 1

data2process = raw_gene_data["data_mat"][selected_genes, :]
gene_ids = []
gene_to = []
gene_name = []
for i in range(len(selected_genes)):
    if selected_genes[i]:
        gene_ids.append(raw_gene_data['probs'][i])
        gene_to.append(
        raw_il["gene_to"][raw_il["gene_from"].index([raw_gene_data['probs'][i]])][0])
        gene_name.append(
            raw_il["gene_name"][raw_il["gene_from"].index([raw_gene_data['probs'][i]])][0])
data2process = data2process.reshape(data2process.shape[1], -1)
scaler = StandardScaler()
scaler.fit(data2process)
X = scaler.transform(data2process)
y_one_hot = classification_utils.convertToOneHot(outcomevar)

kf = KFold(n_splits=FOLDS_NUM)
parN = y_one_hot.shape[0]
kf.get_n_splits(range(parN))
traininds=[]
devinds=[]
testinds=[]



permed_inds = np.random.permutation(range(parN))
for traindev_n, test_n in kf.split(range(parN)):
    testinds.append(permed_inds[test_n])
    testinds[-1] = balanceClusters(testinds[-1], outcomevar)

    kf1 = KFold(n_splits=FOLDS_NUM)
    kf1.get_n_splits(range(len(traindev_n)))


    for train_inds, dev_inds in kf1.split(range(len(traindev_n))):
        traininds.append(permed_inds[traindev_n[train_inds]])
        devinds.append(permed_inds[traindev_n[dev_inds]])
        traininds[-1] = balanceClusters(traininds[-1], outcomevar)
        devinds[-1] = balanceClusters(devinds[-1], outcomevar)
        break



for fold in range(FOLDS_NUM):
    print("fold " + str(fold))
    params = classification_utils.init_model_params()
    params['output_node'] = np.unique(outcomevar).shape[0]
    params['input_node'] = X[traininds[fold], :].shape[1]
    params['batch_size'] = int(np.round(X[traininds[fold], :].shape[0]))
    params["hidden_layers_node"] = [500,300, 100, 50,  20,  5]
    params["lam"] = lam
    params['feature_selection'] = True
    model = Model_param_FS(**params)
    train_acces, train_losses, val_acces, val_losses = model.train(params['param_search'], X[traininds[fold], :],
                                                                   y_one_hot[traininds[fold],:],
                                                                   internal_param[traininds[fold]].reshape(-1, 1),
                                                                   X[devinds[fold], :], y_one_hot[devinds[fold],:],
                                                                   internal_param[devinds[fold]].reshape(-1, 1),
                                                                   '',
                                                                   num_epoch=num_epoch)
    y_pred = model.test(X[traininds[fold], :], internal_param[traininds[fold]])
    acc_tr = metrics.accuracy_score(outcomevar[traininds[fold]], y_pred)
    y_pred = model.test(X[devinds[fold], :], internal_param[devinds[fold]])
    acc_dev = metrics.accuracy_score(outcomevar[devinds[fold]], y_pred)
    y_pred = model.test(X[testinds[fold], :], internal_param[testinds[fold]])
    acc_test = metrics.accuracy_score(outcomevar[testinds[fold]], y_pred)
    print("fold " + str(fold) + "summary")
    print("acc_tr = " + str(acc_tr))
    print("acc_dev = " + str(acc_dev))
    print("acc_test = " + str(acc_test))
    alpha_vals = model.get_prob_alpha(np.array(internal_param[testinds[fold]].reshape(-1, 1)).reshape(-1, 1))
    r_vals = np.array(internal_param[testinds[fold]].reshape(-1, 1)).reshape(-1, 1)
    y = outcomevar[testinds[fold]]
    spio.savemat(respath + "/" + str(Ngenes)  + "_" + str(lam) + "_fold" + str(fold) + ".mat", {'data_labels': gene_name, 'acc_tr': acc_tr, 'acc_dev': acc_dev, 'r_vals': r_vals, 'alpha_vals': alpha_vals, 'y': y, 'y_pred': y_pred})





