import pandas as pd
from sklearn.metrics import auc
from LL4LM.analysis.processing import get_test_accuracies

def running_accuracy(accuracy_matrix):
    dataset_ids = accuracy_matrix.columns.tolist()
    num_datasets = len(dataset_ids)
    interval = len(accuracy_matrix)//num_datasets 
    datasets_seen = []
    running_average = dict()
    for idx, dataset_id in enumerate(dataset_ids):
        datasets_seen.append(dataset_id)
        boundary_idx = ((idx+1) * interval)
        boundary_step = accuracy_matrix.iloc[boundary_idx].name 
        row = accuracy_matrix.loc[boundary_step, datasets_seen]
        running_average[dataset_id] = row.mean()
    running_average = pd.Series(running_average)
    running_average["Average"] = running_average.mean()
    return running_average

def lifelong_auc(accuracy_matrix):
    auc_df = accuracy_matrix.apply(lambda x: auc(accuracy_matrix.index, x)/accuracy_matrix.index[-1], axis=0)
    auc_df["Average"] = auc_df.mean()
    return auc_df

def intransigence_measure(accuracy_matrix, unitask_accuracies):
    dataset_ids = accuracy_matrix.columns.tolist()
    num_datasets = len(dataset_ids)
    interval = len(accuracy_matrix)//num_datasets 
    boundary_accuracy = dict()
    boundary_examples_seen = dict()
    for idx, dataset_id in enumerate(dataset_ids):
        # zero indexing and batch overlap
        boundary_idx = ((idx+1) * interval) - 1 
        row = accuracy_matrix.iloc[boundary_idx]
        boundary_accuracy[dataset_id] = row[dataset_id]
        boundary_examples_seen[dataset_id] = row.name
    boundary_accuracy = pd.Series(boundary_accuracy)
    boundary_examples_seen = pd.Series(boundary_examples_seen)
    intransigence = unitask_accuracies - boundary_accuracy
    intransigence["Average"] = intransigence.mean()
    return intransigence

def forgetting_measure(accuracy_matrix):
    forgetting = (accuracy_matrix.max() - accuracy_matrix.tail(1)).squeeze()
    forgetting["Average"] = forgetting.mean()
    return forgetting

def final_accuracy(accuracy_matrix):
    if isinstance(accuracy_matrix, pd.DataFrame):
        accuracy_matrix = accuracy_matrix.tail(1).squeeze()
    accuracy_matrix["Average"] = accuracy_matrix.mean()
    return accuracy_matrix

def measure_lifelong_metrics(name, stream, logs, multitask_logs, unitask_logs): 
    exp_accuracies = get_test_accuracies(logs, stream)
    mtl_accuracies = get_test_accuracies(multitask_logs, stream)
    utl_accuracies = get_test_accuracies(unitask_logs, stream, unitask=True)
    lfl_auc = lifelong_auc(exp_accuracies).rename(f"{name} AUC")
    mtl_auc = lifelong_auc(mtl_accuracies).rename("Multitask AUC")
    lfl_forgetting = forgetting_measure(exp_accuracies).rename(f"{name} Forgetting")
    mtl_forgetting = forgetting_measure(mtl_accuracies).rename("Multitask Forgetting")
    lfl_intransigence = intransigence_measure(exp_accuracies, utl_accuracies).rename(f"{name} Intransigence")
    mtl_intransigence = intransigence_measure(mtl_accuracies, utl_accuracies).rename("Multitask Intransigence")
    lfl_running_accuracy = running_accuracy(exp_accuracies).rename(f"{name} Running Accuracy")
    lfl_final_accuracy = final_accuracy(exp_accuracies).rename(f"{name} Final Accuracy")
    mtl_final_accuracy = final_accuracy(mtl_accuracies).rename("Multitask Final Accuracy")
    utl_final_accuracy = final_accuracy(utl_accuracies).rename("Unitask Final Accuracy")
    df = pd.concat(
        [lfl_auc, mtl_auc, 
         lfl_forgetting, mtl_forgetting, 
         lfl_intransigence, mtl_intransigence, 
         lfl_running_accuracy, lfl_final_accuracy, mtl_final_accuracy, utl_final_accuracy], 
        axis=1
    )
    return df