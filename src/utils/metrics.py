import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, auc, precision_recall_curve
from mlxtend.evaluate import permutation_test
from scipy import stats
from .upload_metrics import upload_csv_to_sheet
def plot_confusion_matrix(cms: list, classes: list, save_path: str=None, cm_title: str='Confusion Matrix', f1_score=None, classes_order=None):
    """Plot confusion matrix.

    Args:
        cms (list): List of confusion matrices
        classes (list): List of classes
        save_path (str, optional): Path to save the figure. Defaults to None.
        cm_title (str, optional): Title for the confusion matrix. Defaults to 'Confusion Matrix'.
        f1_score (tuple, optional): Tuple containing mean and std of F1 score (mean, std). Defaults to None.
    """

    plt.figure(figsize=(12, 10))
    if len(cms) == 1:
        cm = cms[0]
    else:
        cm = np.mean(cms, axis=0)
        cm_std = np.std(cms, axis=0)
    if classes_order is not None:
        new_order = [classes.index(name) for name in classes_order]
        cm = cm[np.ix_(new_order, new_order)]
        if len(cms) > 1:
            cm_std = cm_std[np.ix_(new_order, new_order)]
        classes = classes_order
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=18)  # Increased fontsize
    plt.yticks(tick_marks, classes, fontsize=18)  # Increased fontsize

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            texto = '{0:.2f}'.format(cm[i, j]) if len(cms) == 1 else '{0:.2f}'.format(cm[i, j]) + '\n$\\pm$' + '{0:.2f}'.format(cm_std[i, j])
            plt.text(j, i, texto,
                     horizontalalignment="center",
                     verticalalignment="center", fontsize=18,  # Increased fontsize
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True', fontsize=20,labelpad=2)  # Increased fontsize
    plt.xlabel('Predicted', fontsize=20,labelpad=2)  # Increased fontsize
    
    # Main title
    plt.title(cm_title, fontsize=20, pad=20)  # Increased fontsize and padding
    
    # Add F1 score as subtitle if provided
    if f1_score is not None:
        f1_mean, f1_std = f1_score
        plt.title(f'{cm_title}\nF1 Score: {f1_mean:.3f} ± {f1_std:.3f}', fontsize=20, pad=20)
    else:
        plt.title(cm_title, fontsize=20, pad=20)  # Increased fontsize and padding

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def print_metrics(acc: tuple, precision: tuple, recall: tuple, f1: tuple,
                save_path: str=None):
    """Print and save metrics.

    Args:
        acc (tuple): Accuracy mean and std
        precision (tuple): Precision mean and std
        recall (tuple): Recall mean and std
        f1 (tuple): F1 mean and std
        roc_auc (tuple): ROC AUC mean and std
        pr_auc (tuple): PR AUC mean and std
        save_path (str, optional): Path to save the metrics. Defaults to None.
    """
    def print_and_save(metric: str, values: tuple, save_path: str=None):
        text = '{}: {:.3f} ± {:.3f}'.format(metric, values[0], values[1])
        print(text)
        if save_path:
            with open(save_path, 'a') as f:
                f.write(text+'\n')

    print_and_save('Accuracy', acc, save_path)
    print_and_save('Precision', precision, save_path)
    print_and_save('Recall', recall, save_path)
    print_and_save('F1', f1, save_path)

def calculate_metrics_logits(targets_list: np.ndarray, preds_list: np.ndarray, logits_target=None, logits_preds=None, path: str=None, 
                     name: str=None, classes: list=None, compare_f1s: list=None, compare_f1s_logits=None, cm_title: str='Confusion Matrix',classes_order= None):
    """Calculate metrics.

    Args:
        targets_list (np.ndarray): True labels
        preds_list (np.ndarray): Predicted labels
        logits_target (np.ndarray, optional): True labels for logits. Defaults to None.
        logits_preds (np.ndarray, optional): Predicted labels for logits. Defaults to None.
        path (str, optional): Path to save the metrics. Defaults to None.
        name (str, optional): Name for saving metrics. Defaults to None.
        classes (list, optional): List of class names. Defaults to None.
        compare_f1s (list, optional): F1 scores from another method to compare against. Defaults to None.
        cm_title (str, optional): Title for confusion matrix. Defaults to 'Confusion Matrix'.
    """
    accs, precs, recs, f1s, cms = [], [], [], [], []

    for targets, preds in zip(targets_list, preds_list):
        accs.append(accuracy_score(targets, preds))
        precs.append(precision_score(targets, preds, average='macro'))
        recs.append(recall_score(targets, preds, average='macro'))
        f1s.append(f1_score(targets, preds, average='macro'))
        cms.append(confusion_matrix(targets, preds, normalize='true'))

    acc = (np.mean(accs), np.std(accs))
    precision = (np.mean(precs), np.std(precs))
    recall = (np.mean(recs), np.std(recs))
    f1 = (np.mean(f1s), np.std(f1s))
    f1_sorted = np.argsort(f1s)
    best_f1 = f1_sorted[-1]


    path_metrics = os.path.join(path, f'{name}_metrics.txt')
    print(path_metrics)
    path_cm = os.path.join(path, f'{name}_cm.png')
    with open(path_metrics, 'a') as f:
    #write all f1-scores
        f.write(f'F1-scores: {f1s} \n')

    if compare_f1s is not None:
        print(f1s)
        print(f'Comparing F1-scores with {compare_f1s}')
        # Using scipy.stats for statistical test between f1 scores
        t_stat, p_value = stats.ttest_ind(f1s, compare_f1s, equal_var=False, alternative='greater')
        # Alternative options:
        # p_value = stats.mannwhitneyu(f1s, compare_f1s)[1]  # Non-parametric test
        # p_value = stats.wilcoxon(f1s, compare_f1s)[1]  # If samples are paired
        print(f'Permutation test p-value: {p_value}')
        with open(path_metrics, 'a') as f:
            #write all f1-scores
            f.write(f'\nPermutation test p-value: {p_value}\n')
            f.write(f'Null hypothesis: no difference between methods\n')
            if p_value < 0.05:
                f.write('Result: Significant difference between methods (p < 0.05)\n')
            else:
                f.write('Result: No significant difference between methods (p >= 0.05)\n')
    
    print_metrics(acc, precision, recall, f1, path_metrics)
    plot_confusion_matrix(cms, classes, path_cm, cm_title, f1_score=f1,classes_order=classes_order)

    # Process logits if provided
    if logits_target is not None and logits_preds is not None:
        logits_accs, logits_precs, logits_recs, logits_f1s, logits_cms = [], [], [], [], []
        
        for targets_l, preds_l in zip(logits_target, logits_preds):
 
            logits_accs.append(accuracy_score(targets_l, preds_l))
            logits_precs.append(precision_score(targets_l, preds_l, average='macro'))
            logits_recs.append(recall_score(targets_l, preds_l, average='macro'))
            logits_f1s.append(f1_score(targets_l, preds_l, average='macro'))
            logits_cms.append(confusion_matrix(targets_l, preds_l, normalize='true'))
        
        logits_acc = (np.mean(logits_accs), np.std(logits_accs))
        logits_precision = (np.mean(logits_precs), np.std(logits_precs))
        logits_recall = (np.mean(logits_recs), np.std(logits_recs))
        logits_f1 = (np.mean(logits_f1s), np.std(logits_f1s))
        
        # Append logits metrics to the same file
        with open(path_metrics, 'a') as f:
            f.write('\n\n----- LOGITS METRICS (Unique Oid) -----\n')
            f.write(f'Logits F1-scores: {logits_f1s} \n')
        
        # Print logits metrics to the same file
        print('\n----- LOGITS METRICS (Unique Oid) -----')
        print_metrics(logits_acc, logits_precision, logits_recall, logits_f1, path_metrics)
        #add permutation test for logits
        if compare_f1s_logits is not None:
            print(f'Comparing Logits F1-scores with {compare_f1s_logits}')
            # Using scipy.stats for statistical test between f1 scores
            t_stat, p_value = stats.ttest_ind(logits_f1s, compare_f1s_logits, equal_var=False, alternative='greater')
            # Alternative options:
            # p_value = stats.mannwhitneyu(logits_f1s, compare_f1s_logits)[1]  # Non-parametric test
            # p_value = stats.wilcoxon(logits_f1s, compare_f1s_logits)[1]  # If samples are paired
            print(f'Permutation test p-value: {p_value}')
            with open(path_metrics, 'a') as f:
                f.write(f'\nPermutation test p-value: {p_value}\n')
                f.write(f'Null hypothesis: no difference between methods\n')
                if p_value < 0.05:
                    f.write('Result: Significant difference between methods (p < 0.05)\n')
                else:
                    f.write('Result: No significant difference between methods (p >= 0.05)\n')
        # Create a separate confusion matrix for logits with "Unique Oid" in the title
        logits_cm_title = f'{cm_title} (by Oid)'
        path_logits_cm = os.path.join(path, f'{name}_unique_oid_cm.png')
        plot_confusion_matrix(logits_cms, classes, path_logits_cm, logits_cm_title, f1_score=logits_f1, classes_order=classes_order)

def calculate_metrics_time(targets_list: np.ndarray, preds_list: np.ndarray,
                                path: str=None, classes: list=None,
                                cm_title: str='Confusion Matrix',
                                classes_order=None, experiment_name=None,
                                all_experiments_csv=None,
                                modality: str=None,
                                max_time_to_eval: int=None):
    """Calculate metrics for different max_time_to_eval values.

    Args:
        targets_list (np.ndarray): True labels
        preds_list (np.ndarray): Predicted labels
        path (str, optional): Path to save the metrics. Defaults to None.
        classes (list, optional): List of class names. Defaults to None.
        cm_title (str, optional): Title for confusion matrix. Defaults to 'Confusion Matrix'.
        classes_order (list, optional): Order of classes for confusion matrix. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Defaults to None.
        all_experiments_csv (str, optional): Path to CSV file for all experiments. Defaults to None.
        modality (str, optional): Modality type (LC, Feat, Mix). Defaults to None.
        max_time_to_eval (int, optional): Maximum time to evaluate. Defaults to None.
    """
    # Derive a name like ATATComparison_modality from the all_experiments_csv path
    if all_experiments_csv is not None:
        base_name = os.path.splitext(os.path.basename(all_experiments_csv))[0]
        all_experiments_csv_name = f"{base_name}_{modality}"
        all_experiments_csv = all_experiments_csv.split('.csv')[0] + f'_{modality}.csv'
    else:
        all_experiments_csv_name = None
        
    f1s, cms = [], []

    for targets, preds in zip(targets_list, preds_list):
        f1s.append(f1_score(targets, preds, average='macro'))
        cms.append(confusion_matrix(targets, preds, normalize='true'))

    f1 = (np.mean(f1s), np.std(f1s))

    path_cm = os.path.join(path, f'{experiment_name}_{modality}_{max_time_to_eval}_cm.pdf')

    # Prepare DataFrame for f1s with mean and std for each max_time_to_eval
    metrics_file = all_experiments_csv
    col_mean = f"{max_time_to_eval}_mean"
    col_std = f"{max_time_to_eval}_std"
    new_data = {
        "experiment_name": experiment_name,
        col_mean: f1[0],
        col_std: f1[1]
    }
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        # Check if experiment_name exists
        if experiment_name in df["experiment_name"].values:
            idx = df.index[df["experiment_name"] == experiment_name][0]
            # Add new columns if they don't exist
            if col_mean not in df.columns:
                df[col_mean] = 0.0
            if col_std not in df.columns:
                df[col_std] = 0.0
            df.at[idx, col_mean] = f1[0]
            df.at[idx, col_std] = f1[1]
        else:
            # Add new row
            for col in df.columns:
                if col not in new_data:
                    new_data[col] = 0.0
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        # Ensure all columns are present
        for col in [col_mean, col_std]:
            if col not in df.columns:
                df[col] = 0.0
        df.to_csv(metrics_file, index=False)
    else:
        df = pd.DataFrame([new_data])
        df.to_csv(metrics_file, index=False)

    upload_csv_to_sheet(all_experiments_csv, all_experiments_csv_name)

    # Plot confusion matrix
    plot_confusion_matrix(cms, classes, path_cm, cm_title, f1_score=f1, classes_order=classes_order)

def calculate_metrics(targets_list: np.ndarray, preds_list: np.ndarray,
                                path: str=None, classes: list=None,
                                baseline_path: str=None, cm_title: str='Confusion Matrix',
                                classes_order=None, experiment_name=None,
                                all_experiments_csv=None,
                                baseline_experiment_name=None,
                                modality: str=None):
    """Calculate metrics without logits.

    Args:
        targets_list (np.ndarray): True labels
        preds_list (np.ndarray): Predicted labels
        path (str, optional): Path to save the metrics. Defaults to None.
        name (str, optional): Name for saving metrics. Defaults to None.
        classes (list, optional): List of class names. Defaults to None.
        compare_f1s (list, optional): F1 scores from another method to compare against. Defaults to None.
        cm_title (str, optional): Title for confusion matrix. Defaults to 'Confusion Matrix'.
        classes_order (list, optional): Order of classes for confusion matrix. Defaults to None.
    """
    baseline_path = baseline_path.split('.csv')[0] + f'_{modality}.csv' if baseline_path is not None else None
    # Derive a name like ATATComparison_modality from the all_experiments_csv path
    if all_experiments_csv is not None:
        base_name = os.path.splitext(os.path.basename(all_experiments_csv))[0]
        all_experiments_csv_name = f"{base_name}_{modality}"
        all_experiments_csv = all_experiments_csv.split('.csv')[0] + f'_{modality}.csv'
    else:
        all_experiments_csv_name = None
    accs, precs, recs, f1s, cms = [], [], [], [], []
    

    for targets, preds in zip(targets_list, preds_list):
        accs.append(accuracy_score(targets, preds))
        precs.append(precision_score(targets, preds, average='macro'))
        recs.append(recall_score(targets, preds, average='macro'))
        f1s.append(f1_score(targets, preds, average='macro'))
        cms.append(confusion_matrix(targets, preds, normalize='true'))

    acc = (np.mean(accs), np.std(accs))
    precision = (np.mean(precs), np.std(precs))
    recall = (np.mean(recs), np.std(recs))
    f1 = (np.mean(f1s), np.std(f1s))


    path_cm = os.path.join(path, f'{experiment_name}_{modality}_cm.pdf')

    # Add experiment_name parameter
    def save_metrics_to_csv(metrics_dict, save_path, experiment_name=None):
        df = pd.DataFrame([metrics_dict])
        if experiment_name is not None:
            df.insert(0, 'experiment_name', experiment_name)
        if not os.path.exists(save_path):
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, mode='a', header=False, index=False)
        print(f"Metrics saved to: {save_path}")

    # Prepare metrics dictionary
    metrics_dict = {
        'Modality': modality,
        'accuracy_mean': acc[0],
        'accuracy_std': acc[1],
        'precision_mean': precision[0],
        'precision_std': precision[1],
        'recall_mean': recall[0],
        'recall_std': recall[1],
        'f1_mean': f1[0],
        'f1_std': f1[1],
    }
    #save f1 to csv where the first row is the experiment name and the other rows are f1s to baseline_path
    if baseline_path is not None:
        # Save f1s as a column with experiment_name as header, each f1 in a new row
        #f1s_dict = {experiment_name: f1s}
        f1s_df = pd.DataFrame(f1s, columns=[experiment_name])
        if os.path.exists(baseline_path):
            existing_df = pd.read_csv(baseline_path)
            existing_df[experiment_name] = pd.Series(f1s)
            existing_df.to_csv(baseline_path, index=False)
        else:
            f1s_df.to_csv(baseline_path, index=False)
        #save_metrics_to_csv(f1s_dict, baseline_path, experiment_name=None)
    


    try:
        compare_f1s = pd.read_csv(baseline_path)[baseline_experiment_name].values if baseline_path is not None else None
    except:
        compare_f1s = None
    if compare_f1s is not None:
        t_stat, p_value = stats.ttest_ind(f1s, compare_f1s, equal_var=False, alternative='greater')
        metrics_dict['permtest_pvalue'] = p_value
        metrics_dict['permtest_significant'] = p_value < 0.05
        metrics_dict['compared_to'] = baseline_experiment_name

    # Save metrics to CSV (append as new row to all_experiments_csv)
    #if all_experiments_csv is not None:
    save_metrics_to_csv(metrics_dict, all_experiments_csv, experiment_name=experiment_name)
    upload_csv_to_sheet(all_experiments_csv, all_experiments_csv_name)
    # save accs, precs, recs, f1s to csv file where columns are metrics and rows are experiments in csv format
    path_metrics = os.path.join(path, f'{experiment_name}_{modality}_metrics.csv')
    metrics_df = pd.DataFrame({
        'accuracy': accs,
        'precision': precs,
        'recall': recs,
        'f1': f1s
    })
    metrics_df.to_csv(path_metrics, index=False)
    

    # Plot confusion matrix as before
    plot_confusion_matrix(cms, classes, path_cm, cm_title, f1_score=f1, classes_order=classes_order)

