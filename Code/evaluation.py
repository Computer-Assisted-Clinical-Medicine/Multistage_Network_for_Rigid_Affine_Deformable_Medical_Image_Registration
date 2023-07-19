import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

from NetworkBasis import config as cfg
import NetworkBasis.util as util
import NetworkBasis.loadsavenii as loadsave
import metric
from processing import seg_binary


def make_csv_header():
    """
        Creates a list representing the header row for a CSV file that contains evaluation metrics.

        Returns:
            header_row (list): The header row for the CSV file.
        """
    if cfg.seg_available:
        header_row = ['FILENAME_FIXED', 'FILENAME_MOVING', 'DICE', 'JCD_NEG', 'JCD_NEG_PERC']
    else:
        header_row = ['FILENAME_FIXED', 'FILENAME_MOVING', 'JCD_NEG', 'JCD_NEG_PERC']

    return header_row

def make_csv_header_eval_fold():
    """
            Creates a list representing the header row for a CSV file that contains evaluation metrics.

            Returns:
                header_row (list): The header row for the CSV file.
            """
    if cfg.seg_available:
        header_row = ['DICE', 'JCD_NEG', 'JCD_NEG_PERC']
    else:
        header_row = ['JCD_NEG', 'JCD_NEG_PERC']
    return header_row

def evaluate_prediction(result_metrics, prediction_path, filename_predict, fixed_img_path, filename_fixed, filename_moving,
                        filename_seg_fixed="None", filename_seg_moving="None"):
    """
        This function evaluates the prediction by calculating various metrics between the predicted image and the fixed image.

        Args:
            result_metrics (dict): A dictionary containing the results of the metrics calculated for the prediction.
            prediction_path (str): The path to the folder containing the predicted image and the associated files.
            filename_predict (str): The filename of the predicted image.
            fixed_img_path (str): The path to the folder containing the fixed image.
            filename_fixed (str): The filename of the fixed image.
            filename_moving (str): The filename of the moving image.
            filename_seg_fixed (str): The filename of the fixed segmentation image, default value is "None".
            filename_seg_moving (str): The filename of the moving segmentation image, default value is "None".

        Returns:
            A dictionary containing the results of the metrics calculated for the prediction.
        """
    if cfg.seg_available:
        fixed_seg_img = sitk.ReadImage(os.path.join(cfg.path_seg_fixed , filename_seg_fixed), sitk.sitkFloat32)
        fixed_seg_np = sitk.GetArrayFromImage(fixed_seg_img)
        fixed_seg_np = np.moveaxis(fixed_seg_np, 0, -1)
        fixed_seg_np = seg_binary(fixed_seg_np)

        moved_seg_img = sitk.ReadImage(os.path.join(prediction_path + "/seg/", "seg_" + filename_seg_moving), sitk.sitkFloat32)
        moved_seg_np = sitk.GetArrayFromImage(moved_seg_img)
        moved_seg_np = np.moveaxis(moved_seg_np, 0, -1)
        moved_seg_np = seg_binary(moved_seg_np)

        result_metrics['DICE'] = metric.dice_coefficient(moved_seg_np, fixed_seg_np)

    path_disp = prediction_path + "/displacementfields/"

    displacementfield = loadsave.load_displacementfield(path_disp, filename_moving)

    result_metrics['JCD_NEG'] = metric.jc(displacementfield)
    result_metrics['JCD_NEG_PERC'] = metric.jc_proz(displacementfield)

    return result_metrics

def combine_evaluation_results_from_folds(experiment_path):
    """
       Combines the evaluation results from different folds (for cross-validation).

       Args:
           experiment_path (str): The path to the directory containing the evaluation results.

       Returns:
           None
       """
    eval_mean_file_path = os.path.join(experiment_path, 'evaluation-mean.csv')
    eval_std_file_path = os.path.join(experiment_path, 'evaluation-std.csv')

    header_row = make_csv_header_eval_fold()
    results = []
    mean_statistics = []
    std_statistics = []

    dir_list = os.listdir(experiment_path)
    for file in dir_list:
        if file.endswith('.csv') and file != 'evaluation-mean.csv' and file != 'evaluation-std.csv'  \
                and file != 'all_data.csv' and file != 'dice_0.csv' and file != 'dice_1.csv' and file != 'dice_2.csv'\
                and file != 'dice_3.csv' and file != 'dice_4.csv':
            print(file)
            try:
                individual_results = pd.read_csv(experiment_path+file, dtype=object,sep=';').values
                #print(individual_results)
            except:
                print('Could not read', file)
            results.append(np.float32(individual_results[:-3,2:]))

    util.make_csv_file(eval_mean_file_path, header_row)
    util.make_csv_file(eval_std_file_path, header_row)

    if len(results) > 0:
        results = np.concatenate(results)

        average_results = np.mean(results, axis=0).tolist()
        mean_statistics.append(average_results)

        std_results = np.std(results, axis=0).tolist()  # column std

        std_statistics.append(std_results)

    for row in mean_statistics:
        with open(eval_mean_file_path, 'a', newline='') as evaluation_file:
            eval_csv_writer = csv.writer(evaluation_file, delimiter=';', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)
            eval_csv_writer.writerow(row)

    for row in std_statistics:
        with open(eval_std_file_path, 'a', newline='') as evaluation_file:
            eval_csv_writer = csv.writer(evaluation_file, delimiter=';', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)
            eval_csv_writer.writerow(row)

def combine_evaluation_results_in_file(experiment_path):
    """
       Combines the evaluation results from different folds (for cross-validation) in one file (all_data.csv).

       Args:
           experiment_path (str): The path to the directory containing the evaluation results.

       Returns:
           None
       """

    all_data_file_path = os.path.join(experiment_path, 'all_data.csv')
    header_row = make_csv_header_eval_fold()
    results = []

    dir_list = os.listdir(experiment_path)
    for file in dir_list:
        if file.endswith('.csv') and file != 'evaluation-mean.csv' and file != 'evaluation-std.csv' \
                and file != 'all_data.csv'and file != 'dice_0.csv' and file != 'dice_1.csv' and file != 'dice_2.csv'\
                and file != 'dice_3.csv' and file != 'dice_4.csv':
            print(file)
            try:
                individual_results = pd.read_csv(experiment_path+file, dtype=object,sep=';').values
            except:
                print('Could not read', file)
            results.append(np.float32(individual_results[:-3,2:]))

    util.make_csv_file(all_data_file_path, header_row)

    if len(results) > 0:
        results = np.concatenate(results)

    for row in results:
        with open(all_data_file_path, 'a', newline='') as evaluation_file:
            eval_csv_writer = csv.writer(evaluation_file, delimiter=';', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)
            eval_csv_writer.writerow(row)

def make_boxplot_graphic(experiment_path):
    """
        Creates a boxplot for each metric, box for each k-fold-eval-csv

        Args:
            experiment_path (str): The path to the csv files containing the evaluation results.

        Returns:
            None
            """
    if not os.path.exists(os.path.join(experiment_path, 'plots')):
        os.makedirs(os.path.join(experiment_path, 'plots'))

    linewidth = 2
    metrics = make_csv_header_eval_fold()

    for title in metrics:
        data = []
        labels = []

        for i in range(cfg.kfold): # compare k-fold results
            eval_file_path = experiment_path + 'eval-'+str(i)+'.csv'
            individual_results = pd.read_csv(eval_file_path, dtype=object, usecols=[title],sep=';',skipfooter=3,engine='python').values
            data.append(np.squeeze(np.float32(individual_results)))
            labels.append('eval-'+str(i)+'.csv')

            f = plt.figure(figsize=(2 * len(data) + 5, 10))
            ax = plt.subplot(111)
            [i.set_linewidth(1) for i in ax.spines.values()]

        # ax.set_title(title, pad=20)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        if any(x in title for x in metrics):
            ax.set_ylim([0, 1])
            ax.set_ylim(auto=True)

        else:
            ax.set_ylabel('mm')
            if 'Hausdorff' in title:
                ax.set_ylim([0, 140])
            elif 'Mean' in title:
                ax.set_ylim([0, 40])

        p=plt.boxplot(data, notch=False, showmeans=True, showfliers=True, vert=True, widths=0.9,
                        patch_artist = True,autorange=True,labels=labels)

        if "boxes" in p:
            [box.set_color(_get_color("-")) for label, box in zip(labels, p["boxes"])]
            [box.set_facecolor(_get_color(label)) for label, box in zip(labels, p["boxes"])]
            [box.set_linewidth(linewidth) for box in p["boxes"]]
        if "whiskers" in p:
            for label, whisker in zip(np.repeat(labels, 2), p["whiskers"]):
                if str(2) in label:
                    whisker.set_linestyle('dashed')
                else:
                    whisker.set_linestyle('dotted')
                whisker.set_color(_get_color(label))
                whisker.set_linewidth(linewidth)
        if "medians" in p:
            for label, median in zip(labels, p["medians"]):
                median.set_color(_get_color("-"))
                median.set_linewidth(linewidth)
        if "means" in p:
            for label, mean in zip(labels, p["means"]):
                if str(2) in label:
                    mean.set_marker('x')
                else:
                    mean.set_marker('+')
                mean.set_markeredgecolor(_get_color("-"))
                mean.set_markerfacecolor(_get_color("-"))
                mean.set_linewidth(linewidth)
        if "caps" in p:
            [cap.set_color(_get_color(label)) for label, cap in zip(np.repeat(labels, 2), p["caps"])]
            [cap.set_linewidth(linewidth) for cap in p["caps"]]
        if "fliers" in p:
            [flier.set_color(_get_color(label)) for label, flier in zip(labels, p["fliers"])]
            [flier.set_markeredgecolor(_get_color(label)) for label, flier in zip(labels, p["fliers"])]
            [flier.set_markerfacecolor(_get_color(label)) for label, flier in zip(labels, p["fliers"])]

            [flier.set_fillstyle("full") for label, flier in zip(labels, p["fliers"])]

        plt.savefig(os.path.join(experiment_path, 'plots', title.replace(' ', '') + '.png'), transparent=True)

def _get_color(label):
    """
        Get the color associated with a given label.

        Args:
            label: A string representing the label.

        Returns:
            A string representing the color associated with the label.
        """
    colors = {"-": "#000000","eval-0.csv": "#e27978", "eval-1.csv": "#EFBE7D", "eval-2.csv": "#cca9dd",
              "eval-3.csv": "#89CA89", "eval-4.csv": "#6FB7D6"}

    if label in colors:
        return colors[label]
    else:
        return "#808080"

