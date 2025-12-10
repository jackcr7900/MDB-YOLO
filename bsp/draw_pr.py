import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_list = ['D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue_190.csv',
                 # 'D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue147.csv',
                 'D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue_191.csv',]
    name = ['YOLOv8s', 'AIT-YOLOv8']
    ap = ['0.864',  '0.896']

    plt.figure(figsize=(6,6))
    for i in range(len(file_list)):
        pr_data = pd.read_csv(file_list[i], header=None)
        recall, precision = np.array(pr_data[0]), np.array(pr_data[1])

        plt.plot(recall, precision, label=name[i] + ' mAP@50: ' + ap[i])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pr.png')

# def plot_metrics(ax, metric_col_name, y_label, color, modelname, is_pr=False):
#     res_path = pr_csv_dict[modelname]
#     try:
#         data = pd.read_csv(res_path)
#         data.columns = data.columns.str.strip()  # Remove spaces from column names
#
#         if is_pr:
#             precision = data['metrics/precision(B)'].values
#             recall = data['metrics/recall(B)'].values
#             ax.plot(recall, precision, label=modelname, color=color, linewidth='2')  # Set color and linewidth for PR curve
#         else:
#             epochs = data['epoch'].values  # epoch column
#             metric_data = data[metric_col_name].values  # Get the corresponding metric column
#             ax.plot(epochs, metric_data, label=modelname, color=color, linewidth='2')
#
#     except Exception as e:
#         print(f"Error reading {modelname}: {e}")
#
# # Main function
# def plot_all_metrics():
#     global pr_csv_dict
#     pr_csv_dict = {
#         'D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue121.csv',
#         'D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue147.csv',
#         'D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue119.csv',
#     }
#
#     colors = {
#         'YOLOv8m': '#00EE76',
#         'YOLOv8s': '#EEEE00',
#         'AIT-YOLOv8': '#8470FF',
#     }
#
#     fig, axs = plt.subplots(1, 3, figsize=(24, 8), tight_layout=True)  # 1 row, 3 columns
#
#     # Set global font size
#     plt.rcParams.update({'font.size': 16})
#
#     # Plot PR Curve
#     file_list = ['D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue121.csv',
#                      'D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue147.csv',
#                      'D:\\Yolov\\ultralytics-8.2.0\\bsp\\pr\\Taro_residue119.csv',]
#     names = ['YOLOv8s', 'YOLOv8m', 'AIT-YOLOv8']
#
#     for i in range(len(file_list)):
#         pr_data = pd.read_csv(file_list[i], header=None)
#         recall, precision = np.array(pr_data[0]), np.array(pr_data[1])
#         color = colors[f'{names[i]}']  # Use the corresponding color
#         axs[0].plot(recall, precision, label=f'{names[i]}', color=color, linewidth='2')  # Set linewidth
#
#     axs[0].set_xlabel('Recall', fontsize=16)
#     axs[0].set_ylabel('Precision', fontsize=16)
#     axs[0].set_xlim(0, 1)
#     axs[0].set_ylim(0, 1)
#     axs[0].legend(loc='lower right', fontsize=16)
#     axs[0].spines['top'].set_linewidth(2)
#     axs[0].spines['right'].set_linewidth(2)
#     axs[0].spines['left'].set_linewidth(2)
#     axs[0].spines['bottom'].set_linewidth(2)
#     axs[0].tick_params(width=2, labelsize=14)
#     axs[0].set_title('Precision-Recall Curve', fontsize=18)
#
#     # Plot mAP@0.5
#     for modelname in pr_csv_dict:
#         plot_metrics(axs[1], 'metrics/mAP50(B)', 'mAP@0.5', colors[modelname], modelname)
#
#     axs[1].set_xlabel('Epoch', fontsize=16)
#     axs[1].set_ylabel('mAP@0.5', fontsize=16)
#     axs[1].set_xlim(0, None)
#     axs[1].set_ylim(0, 1)
#     axs[1].legend(loc='lower right', fontsize=16)
#     axs[1].spines['top'].set_linewidth(2)
#     axs[1].spines['right'].set_linewidth(2)
#     axs[1].spines['left'].set_linewidth(2)
#     axs[1].spines['bottom'].set_linewidth(2)
#     axs[1].tick_params(width=2, labelsize=14)
#     axs[1].set_title('mAP@0.5', fontsize=18)
#
#     # Plot mAP@0.95
#     for modelname in pr_csv_dict:
#         plot_metrics(axs[2], 'metrics/mAP50-95(B)', 'mAP@0.95', colors[modelname], modelname)
#
#     axs[2].set_xlabel('Epoch', fontsize=16)
#     axs[2].set_ylabel('mAP@0.95', fontsize=16)
#     axs[2].set_xlim(0, None)
#     axs[2].set_ylim(0, 1)
#     axs[2].legend(loc='lower right', fontsize=16)
#     axs[2].spines['top'].set_linewidth(2)
#     axs[2].spines['right'].set_linewidth(2)
#     axs[2].spines['left'].set_linewidth(2)
#     axs[2].spines['bottom'].set_linewidth(2)
#     axs[2].tick_params(width=2, labelsize=14)
#     axs[2].set_title('mAP@0.95', fontsize=18)
#
#     plt.subplots_adjust(wspace=0.3)  # Adjust spacing between subplots
#
#     # Save the figure
#     plt.savefig('/yolo11/yolo11-1/images/aa/diff_yolo_metrics6.png', dpi=250)#保存位置
#     plt.show()
#
# # Execute plotting
# if __name__ == '__main__':
#     plot_all_metrics()
