from ultralytics import YOLO
import torch
import yaml


def convert_pt_to_onnx():
    # 加载训练好的模型
    model = YOLO('D:\\Yolov\\ultralytics-8.2.0\\bsp\\runs\\detect\\train205\\weights\\best.pt')  # 加载您的模型文件

    # 设置导出配置
    export_config = {
        'imgsz': [640, 640],  # 输入图像尺寸
        'batch': 1,  # batch size
        'device': 0,  # 使用GPU，如果是CPU则设为'cpu'
        'simplify': True,  # 简化ONNX模型
        'opset': 12,  # ONNX算子集版本
        'half': False,  # 是否使用FP16
    }

    # 导出模型
    success = model.export(format='onnx', **export_config)
    if success:
        print('模型转换成功！输出文件为: best.onnx')

        # 生成ONNX模型的.yaml文件
        onnx_yaml_config = {
            'type': 'yolov8',
            'name': 'MyModel',  # 模型名称
            'display_name': 'MyCS',  # 显示名称
            'model_path': 'best.onnx',  # ONNX模型路径
            'input_width': export_config['imgsz'][0],  # 输入图像宽度
            'input_height': export_config['imgsz'][1],  # 输入图像高度
            'stride': 32,  # 步幅
            'nms_threshold': 0.45,  # 非极大值抑制阈值
            'confidence_threshold': 0.45,  # 置信度阈值
            'classes': list(model.names.values())  # 类别标签
        }

        # 保存.yaml文件
        yaml_file_path = 'D:\\Yolov\\ultralytics-8.2.0\\bsp\\runs\\detect\\train205\\weights\\best_onnx_config.yaml'
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(onnx_yaml_config, yaml_file, allow_unicode=True)
        print(f'ONNX配置文件已生成: {yaml_file_path}')
    else:
        print('模型转换失败！')


if __name__ == '__main__':
    convert_pt_to_onnx()